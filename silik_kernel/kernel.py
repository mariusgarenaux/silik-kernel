# Basic python dependencies
import os
from pathlib import Path
import traceback
from uuid import uuid4, UUID
from typing import Literal, Optional, Tuple

# Internal dependencies
from .tools import (
    ALL_KERNELS_LABELS,
    setup_kernel_logger,
    KernelMetadata,
    KernelTreeNode,
    SilikCommand,
)

from .types import (
    ExecutionResult,
    IOPubMsg,
    JupyterMessage,
    ExecuteRequestContent,
    ErrorContent,
    ExecuteResultContent,
)

# External dependencies
from ipykernel.kernelbase import Kernel
from jupyter_client.multikernelmanager import MultiKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from statikomand import KomandParser

SILIK_VERSION = "1.5.1"


class SilikBaseKernel(Kernel):
    """
    Silik Kernel - Multikernel Manager
    Silik kernel is MultiKernelManager, wrapped in a jupyter kernel.

    It is a gateway that distributes code cells towards sub-kernels, e.g. :

    - octave
    - pydantic ai agent based kernel (https://github.com/mariusgarenaux/pydantic-ai-kernel)
    - python
    - ir
    - an other silik-kernel !
    - ...

    See https://github.com/Tariqve/jupyter-kernels for available
    kernels.

    The silik-kernel makes basic operations to properly start and
    stop sub-kernels, as well as providing helper functions to distribute
    code to sub-kernels through appropriate jupyter kernel channels.

    You can subclass this kernel in order to define custom strategies
    for :
        - sending messages to sub-kernels
        - merging outputs and errors of sub-kernels outputs
        - sending context (input and outputs of cells) to sub-kernels

    For example, you can implement a custom algorithm that makes
    a majority vote between several chatbot-kernels outputs, or
    create a workflow between kernels. You can also create a dynamic
    strategy that sends code to only one kernel, and share output
    with all sub-kernels.
    """

    implementation = "Silik"
    implementation_version = SILIK_VERSION
    language = "no-op"
    language_version = SILIK_VERSION
    language_info = {
        "name": "silik",
        "mimetype": "text/plain",
        "file_extension": ".txt",
    }
    banner = "Silik Kernel - Multikernel Manager - Run `help` for commands"
    all_kernels_labels: list[str] = ALL_KERNELS_LABELS
    mkm: MultiKernelManager = MultiKernelManager()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_label_rank = 1
        self.kernel_metadata = KernelMetadata(
            self.all_kernels_labels[0], "silik", self.ident
        )
        should_custom_log = os.environ.get("SILIK_KERNEL_LOG", "False")
        should_custom_log = (
            True if should_custom_log in ["True", "true", "1"] else False
        )

        if should_custom_log:
            logger = setup_kernel_logger(__name__, self.kernel_metadata.id)
            logger.debug(f"Started kernel {self.kernel_metadata} and initalized logger")
            self.logger = logger
        else:
            self.logger = self.log

        self.ksm = KernelSpecManager()
        self.mode: Literal["cmd", "cnct"] = "cmd"
        self.active_kernel: KernelMetadata = self.kernel_metadata
        self.root_node: KernelTreeNode = KernelTreeNode(
            self.kernel_metadata
        )  # stores the tree of all kernels
        self.all_kernels: list[KernelMetadata] = []

        self.init_commands()

        self.given_labels = [self.kernel_metadata.label]

    # ------------------------------------------------------ #
    # ------------- ipykernel wrapper methods -------------- #
    # ------------------------------------------------------ #

    def do_execute(  # pyright: ignore
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> ExecutionResult:
        """
        Executes code on this kernel, according to 2 modes :
            - command mode (/cmd),
            - connection mode (/cnct).

        The first one is used to spawn and configure new kernels.
        The second is used to directly connect input of this kernel
        to sub-kernel ones, and to display their output.

        Parameters:
        ---
            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g. for Python’s raw_input()).

        Returns:
        ---
            ExecutionResult, according to Jupyter documentation.
        """

        # first checks for mode switch trigger (command // connect)
        try:
            first_word_trigger = code.split(" ")[0]
            if first_word_trigger in ["/cmd", "/cnct"]:
                self.logger.debug("Detected switch mode trigger")
                if first_word_trigger == "/cmd":
                    self.mode = "cmd"
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": self.execution_count,
                            "data": {
                                "text/plain": "Command mode. You can create and select kernels. Send `help` for the list of commands."
                            },
                            "metadata": {},
                        },
                    )
                    return {
                        "status": "ok",
                        "execution_count": self.execution_count,
                        "payload": [],
                        "user_expressions": {},
                    }
                else:
                    self.mode = "cnct"
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": self.execution_count,
                            "data": {
                                "text/plain": f"All cells are executed on kernel {self.active_kernel.label} [{self.active_kernel.type}]. Run /cmd to exit this mode and select a new kernel."
                            },
                            "metadata": {},
                        },
                    )

                    return {
                        "status": "ok",
                        "execution_count": self.execution_count,
                        "payload": [],
                        "user_expressions": {},
                    }

            # then either run code, or give it to sub-kernels
            if self.mode == "cmd":
                self.logger.debug("Running command")
                execution_result, msg = self.do_execute_on_silik(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                if execution_result.get("status") == "error":
                    self.send_response(self.iopub_socket, "error", msg)
                else:
                    self.send_response(self.iopub_socket, "execute_result", msg)
                self.logger.debug("Output of command : {msg}")
                return execution_result
            elif self.mode == "cnct":
                execution_result, msg = self.do_execute_on_sub_kernel(
                    code, silent, store_history, user_expressions, allow_stdin
                )

                if execution_result["status"] == "error":
                    self.send_response(self.iopub_socket, "error", msg["content"])
                elif execution_result["status"] == "ok":
                    if msg["msg_type"] in [
                        "stream",
                        "display_data",
                        "update_display_data",
                        "execute_input",
                        "execute_result",
                        "error",
                        "clear_output",
                        "status",
                    ]:
                        self.send_response(
                            self.iopub_socket, msg["msg_type"], msg["content"]
                        )
                    else:
                        error_content: ErrorContent = {
                            "ename": "NotIOPubMsg",
                            "evalue": "",
                            "traceback": [
                                f"The message from sub-kernel is of type {msg['msg_type']}; which is not known to be sendable through IOPubSocket."
                            ],
                        }
                        self.send_response(self.iopub_socket, "error", error_content)
                        return {
                            "execution_count": self.execution_count,
                            "payload": [],
                            "status": "error",
                            "user_expressions": {},
                        }

                return execution_result
            else:
                self.mode = "cmd"
                return {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }
        except Exception as e:
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "SilikExecutionError",
                    "evalue": "",
                    "traceback": traceback.format_exception(e),
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

    def do_execute_on_silik(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> Tuple[ExecutionResult, IOPubMsg]:
        """
        Do execute method for "command mode". Runs command on silik kernel.
        Commands can be multiline commands. Each line itself can be splitted
        between | for 'bash like' pipes.

        Parameters:
        ---
            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase
                 the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after
                the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g.
                for Python’s raw_input()).

        Returns :
        ---
            A tuple (ExecutionResult, IOPubMsg). The execution result is the expected output
            of do_execute method of IPykernel wrapper. The IOPubMsg is the message that
            is meant to be sent to IOPub Socket.
        """
        splitted_code = code.split("\n")
        self.logger.debug(f"Splitted code {splitted_code}")
        if len(splitted_code) == 1:
            return self.do_execute_one_command_on_silik(
                code, silent, store_history, user_expressions, allow_stdin
            )
        execution_result, msg = None, None
        for line in splitted_code:
            execution_result, msg = self.do_execute_one_command_on_silik(
                line, True, store_history, user_expressions, allow_stdin
            )
            if execution_result.get("status") == "error":
                return execution_result, msg
        if execution_result is None or msg is None:
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }, {
                "ename": "UnknownCommand",
                "evalue": "Unknown command",
                "traceback": ["Internal Error"],
            }
        return execution_result, msg

    def do_execute_one_command_on_silik(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
        cmd_stdin: str | None = None,
    ) -> Tuple[ExecutionResult, IOPubMsg]:
        """
        Execute one of command in silik. Do not send result to iopub socket;
        but returns the message that is meant to be send to iopub socket,
        as well as the execution result.

        Made to take into account pipes, >, ...

        Parameters :
        ---
            - code (str) : the line of code that will be executed
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase
                 the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after
                the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g.
                for Python’s raw_input()).
            - cmd_stdin : None or str. If str, it represent a positional
                argument that is meant to be given to the command parser.

        Returns :
        ---
            A tuple (ExecutionResult, IOPubMsg). The execution result is the expected output
            of do_execute method of IPykernel wrapper. The IOPubMsg is the message that
            is meant to be sent to IOPub Socket.
        """
        splitted = code.split(maxsplit=1)
        if len(splitted) == 0:
            self.logger.warning(f"Code {code} is empty")
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }, {
                "ename": "UnknownCommand",
                "evalue": "Unknown command",
                "traceback": ["Could not parse command"],
            }
        cmd_name = splitted[0]
        self.logger.debug(f"Splitted command. Command name : `{cmd_name}`.")
        if cmd_name not in self.all_cmds:
            self.logger.warning(f"Command `{cmd_name}` was not found")

            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }, {
                "ename": "UnknownCommand",
                "evalue": cmd_name,
                "traceback": [
                    f"Command `{cmd_name}` not found. Available commands : {self.all_cmds.keys()}"
                ],
            }
        cmd_obj = self.all_cmds[cmd_name]
        if len(splitted) <= 1:
            self.logger.debug(f"No arguments were found for command {cmd_name}")
            args_str = ""
        else:
            args_str = splitted[1]
            self.logger.debug(f"Arguments for command `{cmd_name}` : `{args_str}`")

        args = cmd_obj.parser.parse(args_str)
        self.logger.debug(f"Parsed arguments of `{cmd_name}` : `{vars(args)}`")
        try:
            execution_result, msg = cmd_obj.handler(args)
        except Exception as e:
            self.logger.info(f"Error when running command `{cmd_name}` : `{e}`")
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }, {
                "ename": "CommandError",
                "evalue": str(e),
                "traceback": traceback.format_exception(e),
            }

        self.logger.debug(f"Output of `{cmd_name}` : `{execution_result}`, `{msg}`")
        return execution_result, msg

    # TODO : reimplement properly the pipe (|)
    #       -> need to define properly what is the stdin input of a silik command :-)

    # def do_execute_one_line_on_silik(
    #     self,
    #     code: str,
    #     silent: bool,
    #     store_history: bool = True,
    #     user_expressions: Optional[dict] = None,
    #     allow_stdin: bool = False,
    # ) -> Tuple[ExecutionResult, IOPubMsg]:
    #     """
    #     Executes code on this kernel, without giving it to sub kernels.
    #     Sends content to IOPub channel. Accepted code contains bash like
    #     commands, that is parsed and executed according to self.all_cmds
    #     dictionnary.

    #     Example commands actions :
    #         - display tree,
    #         - display active sub-kernels,
    #         - select kernel to run future code on,
    #         - start a kernel,
    #         - running one-line code on sub kernels.

    #     Parameters:
    #     ---

    #         - code (str) : The code to be executed.
    #         - silent (bool) : Whether to display output.
    #         - store_history (bool) : Whether to record this code in history and increase the execution count. If silent is True, this is implicitly False.
    #         - user_expressions (dict) : Mapping of names to expressions to evaluate after the code has run. You can ignore this if you need to.
    #         - allow_stdin (bool) : Whether the frontend can provide input on request (e.g. for Python’s raw_input()).

    #     Returns :
    #     ---
    #         A tuple (ExecutionResult, IOPubMsg). The execution result is the expected output
    #         of do_execute method of IPykernel wrapper. The IOPubMsg is the message that
    #         is meant to be sent to IOPub Socket.
    #     """
    #     lexer = shlex.shlex(code, posix=True, punctuation_chars="|")
    #     lexer.whitespace_split = True
    #     tokens = list(lexer)
    #     self.logger.debug(tokens)

    #     # split between pipes
    #     commands = []
    #     current = []
    #     for tok in tokens:
    #         if tok == "|":
    #             commands.append(current)
    #             current = []
    #         else:
    #             current.append(tok)
    #     if current:
    #         commands.append(current)
    #     self.logger.debug(f"Commands : {commands}")

    #     cmd_stdin = None
    #     execution_result, msg = None, None
    #     for each_command in commands:
    #         execution_result, msg = self.do_execute_one_command_on_silik(
    #             each_command,
    #             silent,
    #             store_history,
    #             user_expressions,
    #             allow_stdin,
    #             cmd_stdin,
    #         )
    #         if execution_result.get("status") == "error":
    #             return execution_result, msg

    #         self.logger.debug(f"CMD out : {msg}, {execution_result}")

    #         if execution_result.get("status") == "ok":
    #             cmd_stdin = msg.get("data", {}).get("text/plain", None)
    #         else:
    #             cmd_stdin = None

    #         self.logger.debug(execution_result, msg)
    #         self.logger.debug(f"stdin : {cmd_stdin}")
    #     if execution_result is None or msg is None:
    #         return {
    #             "status": "error",
    #             "execution_count": self.execution_count,
    #             "payload": [],
    #             "user_expressions": {},
    #         }, {
    #             "ename": "UnknownCommand",
    #             "evalue": "Unknown command",
    #             "traceback": ["Silik Internal Error"],
    #         }

    #     return execution_result, msg

    def do_execute_on_sub_kernel(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> Tuple[ExecutionResult, JupyterMessage]:
        """
        Transfer code to the selected sub-kernel. And sends the content of
        sub-kernel ouput through IOPub channel.

        The parameters are those asked by the IPykernel wrapper method
        do_execute :
        https://jupyter-client.readthedocs.io/en/stable/wrapperkernels.html#MyKernel.do_execute

        Parameters
        ---

            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g. for Python’s raw_input()).

        Returns :
        ---
            A tuple (ExecutionResult, JupyterMessage). The execution result is the expected output
            of do_execute method of IPykernel wrapper. The JupyterMessage is the message from
            the jupyter kernel protocol. Straight from sub-kernel.
        """
        self.logger.debug(f"Code is sent to selected kernel : {self.active_kernel}")

        execution_result, msg = self.send_code_to_sub_kernel(
            self.active_kernel,
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
        )
        self.logger.debug(f"Output of cell : {execution_result, msg}")

        # custom for silent execution : TODO clean display :-)
        # if silent:
        #     return execution_result, msg
        # all_out = []
        # for each_output_type in msg:
        #     # sends content to each channel (error, execution result, ...)
        #     self.logger.debug(f"Output type {msg[each_output_type]}")
        #     if each_output_type == "execute_result":
        #         msg[each_output_type]["data"]["text/plain"] = (
        #             f"{self.active_kernel.label} [{self.active_kernel.type}]\n"
        #             + msg[each_output_type]["data"]["text/plain"]
        #         )
        #     all_out.append(msg[each_output_type])

        return execution_result, msg

    def do_is_complete(self, code: str):
        if self.mode == "cnct":
            km = self.mkm.get_kernel(self.active_kernel.id)
            self.logger.debug(f"Sending is_complete to {self.active_kernel}")
            kc = km.client()
            kc.start_channels()
            msg = kc.session.msg(
                "is_complete_request",
                {
                    "code": code,
                },
            )
            kc.shell_channel.send(msg)
            msg_id = msg["header"]["msg_id"]
            output = {}

            while True:
                msg = kc.get_shell_msg()
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]

                if msg_type == "is_complete_reply":
                    output = msg["content"]
                    break

                elif msg_type == "error":
                    output = msg["content"]
                    break
            self.logger.debug(f"is complete from {self.active_kernel.label}: {output}")
            kc.stop_channels()
            if len(output) == 0:
                return {"status": "unknown"}
            return output
        if code.endswith(" "):
            return {"status": "incomplete", "indent": ""}
        return {"status": "unknown"}

    def do_complete(self, code: str, cursor_pos: int):
        """
        Tab completion. Two modes :
            - cnct : gateway towards tab completion of sub kernel
            - cmd : complete command names from self.all_cmds. TODO :
                implement tab completion for each command args and
                return it.
        """
        try:
            if self.mode == "cnct":
                # just act as a gateway towards active kernel
                km = self.mkm.get_kernel(self.active_kernel.id)
                self.logger.debug(f"Sending do_complete to {self.active_kernel}")
                kc = km.client()
                kc.start_channels()
                msg_type = "complete_request"

                msg = kc.session.msg(
                    "complete_request",
                    {"code": code, "cursor_pos": cursor_pos},
                )
                kc.shell_channel.send(msg)
                msg_id = msg["header"]["msg_id"]
                output = {}

                while True:
                    msg = kc.get_shell_msg()
                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue

                    msg_type = msg["msg_type"]

                    if msg_type == "complete_reply":
                        output = msg["content"]
                        break

                    elif msg_type == "error":
                        output = msg["content"]
                        break
                self.logger.debug(
                    f"do complete from {self.active_kernel.label}: {output}"
                )
                kc.stop_channels()
                if len(output) > 0:
                    return output
            if self.mode == "cmd":
                splitted = code.split(" ")
                self.logger.debug(f"do complete , splitted = {splitted}")
                if len(splitted) == 0:
                    return {
                        "status": "ok",
                        "matches": [],
                        "cursor_start": cursor_pos,
                        "cursor_end": cursor_pos,
                        "metadata": {},
                    }
                if len(splitted) == 1:
                    return self.complete_first_word(splitted, cursor_pos)

                if cursor_pos != len(code):
                    return {
                        "status": "ok",
                        "matches": [],
                        "cursor_start": cursor_pos,
                        "cursor_end": cursor_pos,
                        "metadata": {},
                    }

                if splitted[0] in self.all_cmds:
                    cmd_name = splitted[0]
                    word_to_complete = splitted[-1]
                    self.logger.debug(
                        f"Completing command {cmd_name} | {word_to_complete}"
                    )
                    all_matches = self.all_cmds[cmd_name].parser.do_complete(
                        word_to_complete
                    )

                    return {
                        "status": "ok",
                        "matches": all_matches,
                        "cursor_start": cursor_pos - len(word_to_complete),
                        "cursor_end": cursor_pos,
                        "metadata": {},
                    }
                if splitted[0] not in self.all_cmds:
                    return {
                        "status": "ok",
                        "matches": [],
                        "cursor_start": cursor_pos,
                        "cursor_end": cursor_pos,
                        "metadata": {},
                    }

            return {
                # status should be 'ok' unless an exception was raised during the request,
                # in which case it should be 'error', along with the usual error message content
                # in other messages.
                "status": "ok",
                # The list of all matches to the completion request, such as
                # ['a.isalnum', 'a.isalpha'] for the above example.
                "matches": [],
                # The range of text that should be replaced by the above matches when a completion is accepted.
                # typically cursor_end is the same as cursor_pos in the request.
                "cursor_start": cursor_pos,
                "cursor_end": cursor_pos,
                # Information that frontend plugins might use for extra display information about completions.
                "metadata": {},
            }
        except Exception as e:
            self.logger.warning(traceback.format_exception(e))
            return {
                # status should be 'ok' unless an exception was raised during the request,
                # in which case it should be 'error', along with the usual error message content
                # in other messages.
                "status": "error",
                # The list of all matches to the completion request, such as
                # ['a.isalnum', 'a.isalpha'] for the above example.
                "matches": [],
                # The range of text that should be replaced by the above matches when a completion is accepted.
                # typically cursor_end is the same as cursor_pos in the request.
                "cursor_start": cursor_pos,
                "cursor_end": cursor_pos,
                # Information that frontend plugins might use for extra display information about completions.
                "metadata": {},
            }

    def do_shutdown(self, restart: bool):
        """
        Shutdown the kernel by simply shutting down all sub kernels
        started.
        """
        self.mkm.shutdown_all()
        return {"status": "ok", "restart": restart}

    # ------------------------------------------------------ #
    # ------------- tools for executing code --------------- #
    # ------------------------------------------------------ #

    def complete_first_word(self, splitted_code: list[str], cursor_pos: int):
        first_word = splitted_code[0]
        all_matches = []
        for each_cmd in self.all_cmds:
            if len(each_cmd) < len(first_word):
                continue
            potential_match = each_cmd[: len(first_word)]
            if potential_match == first_word:
                all_matches.append(each_cmd)
        return {
            # status should be 'ok' unless an exception was raised during the request,
            # in which case it should be 'error', along with the usual error message content
            # in other messages.
            "status": "ok",
            # The list of all matches to the completion request, such as
            # ['a.isalnum', 'a.isalpha'] for the above example.
            "matches": all_matches,
            # The range of text that should be replaced by the above matches when a completion is accepted.
            # typically cursor_end is the same as cursor_pos in the request.
            "cursor_start": cursor_pos - len(first_word),
            "cursor_end": cursor_pos,
            # Information that frontend plugins might use for extra display information about completions.
            "metadata": {},
        }

    def get_kernel_history(self, kernel_id: UUID | str) -> list:
        """
        Returns the history of the kernel with kernel_id. Not all kernels
        store their history. See https://jupyter-client.readthedocs.io/en/stable/messaging.html#msging-history
        and https://jupyter-client.readthedocs.io/en/stable/wrapperkernels.html#MyCustomKernel.do_history

        Parameters:
        ---

            - kernel_id (UUID or str): the uuidv4 of kernel

        Returns:
        ---

            A list of 3-tuples, either:
              - (session, line_number, input) or
              - (session, line_number, (input, output)),
            depending on whether output was False or True, respectively.
        """
        if isinstance(kernel_id, UUID):
            kernel_id = str(kernel_id)
        self.logger.debug(f"Getting history for {kernel_id}")
        kc = self.mkm.get_kernel(kernel_id).client()

        try:
            kc.start_channels()

            # Send history request
            msg_id = kc.history(
                raw=True,
                output=False,
                hist_access_type="range",
                session=0,
                start=1,
                stop=1000,
            )

            # Wait for reply
            while True:
                msg = kc.get_shell_msg()
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                if msg["msg_type"] == "history_reply":
                    # history = msg["content"]["history"]
                    self.logger.debug(f"Kernel history : {msg['content']}")
                    return msg["content"]["history"]
        finally:
            kc.stop_channels()

    @property
    def get_available_kernels(self) -> list[str]:
        """
        Finds all the available kernel for this session.
        Same as running in a terminal `jupyter kernelspec list`
        """
        specs = self.ksm.find_kernel_specs()
        return list[str](specs.keys())

    def get_kernel_from_label(self, kernel_label: str) -> KernelMetadata | None:
        """
        Finds a kernel in the tree from its label.
        """
        for each_kernel in self.all_kernels:
            if each_kernel.label == kernel_label:
                return each_kernel

    def find_node_by_metadata(
        self, kernel_metadata: KernelMetadata
    ) -> KernelTreeNode | None:
        """
        Find the tree node (containing tree information like children, parents, ...)
        of a Kernel from its KernelMetadata.

        Parameters:
        ---

            - kernel_metadata (KernelMetadata): the metadata describing the kernel

        Returns:
        ---

            The node object (KernelTreeNode), or None if no match was found
        """

        def recursively_find_node_in_tree(
            node: KernelTreeNode, kernel_metadata: KernelMetadata
        ):
            """
            Recursive method that seeks for a match in children of
            a KernelTreeNode.

            Parameters:
            ---

                - node (KernelTreeNode): the node on which we will search
                    for matching KernelMetadata in its childrens
                - kernel_metadata (KernelMetadata): the metadata of the kernel
                    we need to find the node
            """
            if node.value == kernel_metadata:
                return node

            # Recursively search through children
            for child in node.children:
                found_node = recursively_find_node_in_tree(child, kernel_metadata)
                if found_node:
                    return found_node  # Return the found node if it exists

            return (
                None  # Return None if the target metadata is not found in the subtree
            )

        return recursively_find_node_in_tree(self.root_node, kernel_metadata)

    def find_kernel_metadata_from_path(self, path: str) -> KernelMetadata | None:
        """
        Finds a kernel metadata from its path starting from the active kernel (selected)

        Parameters:
        ---

            - path: the relative path in posix fashion (e.g. kernel_1/kernel_2
                or ../kernel_1/kernel_2) from the active kernel

        Returns:
        ---
            The KernelMetadata located at <path> from active kernel; or None if
            path is incorrect.
        """
        path_components = path.split("/")
        if path_components[-1] == "":
            path_components.pop(-1)

        current_node = self.find_node_by_metadata(self.active_kernel)
        if current_node is None:
            return
        self.logger.debug(
            f"Path components {path_components}, from node {current_node}"
        )
        for component in path_components:
            if component == "..":
                # Move up to the parent node
                current_node = (
                    current_node.parent if current_node.parent else current_node
                )
            else:
                # Find the child node with the corresponding label
                found = False
                for child in current_node.children:
                    if child.value.label == component:
                        current_node = child
                        found = True
                        break

                if not found:
                    return None  # Return None if the path does not exist

        return current_node.value  # Return the KernelMetadata of the found node

    def start_kernel(
        self, kernel_name: str, kernel_label: str | None = None
    ) -> KernelMetadata | None:
        """
        Starts a kernel from its name / type (python3, octave, ...). If no label
        is given, a random one is chosen from a (small list).

        Parameters:
        ---

            - kernel_name (str): the name or type of the kernel (e.g. python3).
                Run `jupyter kernelspec list` in a terminal to get the full list,
                or send `kernels` on a silik kernel ;)
            - kernel_label (str): the label that will be assigned to this kernel,
                internal to silik

        Returns:
        ---
            The KernelMetadata object describing the kernel. None if the label already
            exists.
        """
        self.logger.debug(f"Starting new kernel of type : {kernel_name}")
        kernel_id = str(uuid4())
        if kernel_label is None:
            if self.kernel_label_rank > len(self.all_kernels_labels):
                new_rank = self.kernel_label_rank % len(self.all_kernels_labels)
                specifier = self.kernel_label_rank // len(self.all_kernels_labels)
                kernel_label = self.all_kernels_labels[new_rank] + f"-{specifier}"
                self.kernel_label_rank += 1
            else:
                kernel_label = self.all_kernels_labels[self.kernel_label_rank]
                self.kernel_label_rank += 1
        elif kernel_label in self.given_labels:
            self.logger.debug("Existing label")
            return
        new_kernel = KernelMetadata(label=kernel_label, type=kernel_name, id=kernel_id)
        self.mkm.start_kernel(kernel_name=kernel_name, kernel_id=kernel_id)
        self.given_labels.append(kernel_label)
        self.logger.debug(f"Successfully started kernel {new_kernel}")
        self.logger.debug(f"No kernel with label {kernel_name} is available.")
        return new_kernel

    def send_code_to_sub_kernel(
        self,
        sub_kernel: KernelMetadata,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> Tuple[ExecutionResult, JupyterMessage]:
        """
        Send code to sub kernel for execution through shell channel. Opens a client with
        the kernel, by using MultiKernelManager from JupyterClient.

        Parameters
        ---

            - sub_kernel : A KernelMetadata object describing the sub kernel
            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase the
                execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate
                 after the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request
                 (e.g. for Python’s raw_input()).

        Returns :
        ---
            A tuple (ExecutionResult, JupyterMessage). First element is the expected
            output for do_execute method. The JupyterMessage is the message from
            the jupyter kernel protocol. Straight from sub-kernel.
        """
        km = self.mkm.get_kernel(sub_kernel.id)
        kc = km.client()
        kc.start_channels()
        content: ExecuteRequestContent = {
            "code": code,
            "silent": silent,
            "store_history": store_history,
            "user_expressions": user_expressions,
            "allow_stdin": allow_stdin,
            "stop_on_error": True,
        }
        msg = kc.session.msg(
            "execute_request",
            content,
        )

        kc.shell_channel.send(msg)
        msg_id = msg["header"]["msg_id"]

        while True:
            msg: JupyterMessage = (
                kc.get_iopub_msg()  # pyright: ignore[reportAssignmentType]
            )
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            self.logger.debug(f"Message from sub-kernel : `{msg}`")

            if msg_type in ["execute_result", "stream", "display_data"]:
                kc.stop_channels()
                return {
                    "status": "ok",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }, msg

            if msg_type == "error":
                kc.stop_channels()
                return {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }, msg

            if msg_type == "status" and msg["content"]["execution_state"] == "idle":
                kc.stop_channels()
                return {
                    "status": "ok",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }, msg

        # if sub_kernel.is_branched_to is not None and "execute_result" in output:
        #     raw_output = output["execute_result"]["data"]["text/plain"]
        #     self.logger.debug(
        #         f"Sending output : `{raw_output}` of {sub_kernel.label} to {sub_kernel.is_branched_to.label}"
        #     )
        #     res = self.send_code_to_sub_kernel(
        #         sub_kernel=sub_kernel.is_branched_to,
        #         code=raw_output,
        #         silent=silent,
        #     )
        #     self.logger.debug(res)
        #     return res
        # kc.stop_channels()
        # out_content: ErrorContent = {
        #     "ename": "NotImplementedError",
        #     "evalue": "",
        #     "traceback": [f"Message with id {msg_id} returned a unknown message type."],
        # }
        # return (
        #     {
        #         "status": "error",
        #         "execution_count": self.execution_count,
        #         "payload": [],
        #         "user_expressions": {},
        #     },
        #     JupyterMessage,
        # )

    # ------------------------------------------------------ #
    # ----------------- COMMANDS HANDLERS ------------------ #
    # ------------------------------------------------------ #

    def help_cmd_handler(
        self, args
    ) -> Tuple[
        ExecutionResult, ExecuteResultContent
    ]:  # TODO: retrieve docstrings of handler methods
        doc = {
            "cd <path>": "Moves the selected kernel in the kernel tree",
            "ls | tree": "Displays the kernels tree",
            "mkdir <kernel_type> --label=<kernel_label>": "starts a kernel (see 'kernels' command)",
            "run <code>": "run code on selected kernel - in one shot",
            "restart": "restart the selected kernel",
            "branch <kernel_label>": "branch the output of selected kernel to the input of one of its children. Output of parent kernel is now output of children kernel. (In -> Parent Kernel -> Children Kernel -> Out)",
            "detach": "detach the branch starting from the selected kernel",
            "history": "displays the cells input history for this kernel",
            "kernels": "displays the list of available kernels types",
            "/cnct": "direct connection towards selected kernel : cells will be directly executed on this kernel; except if cell content is '/cmd'",
            "/cmd": "switch to command mode (default one) - exit /cnct mode",
        }
        content = "Silik Kernel allows to manage a group of kernels.\n\n"
        content += f"Start by running `mkdir <kernel_type> --label=my-kernel` with <kernel_type> among {self.get_available_kernels}.\n\n"
        content += "Then, you can run `cd my-kernel` and, `run <code>` to run one shot code in this kernel.\n\n"
        content += "You can also run /cnct to avoid typing `run`. /cmd allows at any time to go back to command mode (navigation and creation of kernels).\n\n"
        content += "Here is a quick reminder of available commands : \n\n"
        for key, value in doc.items():
            content += f"• {key} : {value}\n"

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def cd_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        if args.path is None or not args.path:
            found_kernel = self.kernel_metadata
        else:
            found_kernel = self.find_kernel_metadata_from_path(args.path)

        if found_kernel is None:
            raise ValueError(f"Could not find kernel located at {args.path}")

        self.active_kernel = found_kernel
        content = self.root_node.tree_to_str(self.active_kernel)
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def mkdir_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        active_node = self.find_node_by_metadata(self.active_kernel)

        if active_node is None:
            raise ValueError(
                f"Could not find node in the kernel tree with value {self.active_kernel}"
            )

        if not args.kernel_type:
            raise ValueError(
                f"Please specify a kernel-type among {self.get_available_kernels}"
            )

        new_kernel = self.start_kernel(
            args.kernel_type, None if not args.label else args.label
        )
        if new_kernel is None:
            raise ValueError(
                f"The label {args.label} already exists, please choose an other one."
            )
        active_node.children.append(KernelTreeNode(new_kernel, parent=active_node))
        self.all_kernels.append(new_kernel)
        content = self.root_node.tree_to_str(self.active_kernel)
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def ls_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        content = self.root_node.tree_to_str(self.active_kernel)
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def restart_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        self.mkm.restart_kernel(self.active_kernel.id)
        content = f"Restarted kernel {self.active_kernel.label}"
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def kernels_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        content = self.get_available_kernels

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def history_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        content = self.get_kernel_history(self.active_kernel.id)

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def branch_cmd_handler(self, args):
        raise NotImplementedError("Branch command is not yet implemented.")
        # # connects output of selected kernel to one kernel
        # out_kernel = self.get_kernel_from_label(args.kernel_label)
        # if out_kernel is None:
        #     error = True
        #     content = f"No kernel was found with label `{args.kernel_label}`"
        #     return error, content

        # active_kernel_node = self.find_node_by_metadata(self.active_kernel)
        # if active_kernel_node is None:
        #     error = True
        #     content = (
        #         f"No kernel was found on tree with label {self.active_kernel.label}."
        #     )
        #     return error, content

        # out_kernel_node = self.find_node_by_metadata(out_kernel)
        # if out_kernel_node is None:
        #     error = True
        #     content = f"No kernel was found on tree with label {out_kernel.label}."
        #     return error, content

        # if out_kernel_node not in active_kernel_node.children:
        #     error = True
        #     content = f"Kernel {out_kernel.label} not in {self.active_kernel.label} childrens. Branching is only available from a parent to a children."
        #     return error, content

        # self.active_kernel.is_branched_to = out_kernel

        # trust_level = 0
        # if args.trust_level in ["0", "1", "2", 0, 1, 2]:
        #     trust_level = int(args.trust_level)

        # self.active_kernel.branch_trust_level = trust_level  # pyright: ignore
        # content = self.root_node.tree_to_str(self.active_kernel)
        # return False, content

    def detach_cmd_handler(self, args):
        raise NotImplementedError("Detach command is not yet implemented.")

    def run_cmd_handler(self, args) -> Tuple[ExecutionResult, IOPubMsg]:
        """
        Send a message to the active sub kernel. Returns the result
        """
        execution_result, jupyter_msg = self.do_execute_on_sub_kernel(
            args.cmd, silent=False
        )

        msg_type = jupyter_msg["msg_type"]

        match msg_type:
            case "stream":
                content = jupyter_msg["content"]["text"]
                return execution_result, {
                    "execution_count": self.execution_count,
                    "data": {"text/plain": content},
                    "metadata": {},
                }
            case "display_data" | "update_display_data":
                content = jupyter_msg["content"]["data"]["text/plain"]
                return execution_result, {
                    "execution_count": self.execution_count,
                    "data": {"text/plain": content},
                    "metadata": {},
                }
            case "execute_result" | "error":
                return execution_result, jupyter_msg["content"]
            case "execute_input" | "clear_output" | "status":
                return execution_result, {
                    "execution_count": self.execution_count,
                    "data": {"text/plain": ""},
                    "metadata": {},
                }
            case _:
                raise Exception(
                    f"Message of run command is of type {msg_type}; which is not sendable through IOPubSocket"
                )

    def exit_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        raise NotImplementedError("Exit command is not yet implemented.")

    def source_cmd_handler(self, args):
        path = args.path

        with open(path, "rt", encoding="utf-8") as f:
            code = f.read()

        return self.do_execute_on_silik(code, False)

    def cat_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        path = args.path
        with open(path, "rt", encoding="utf-8") as f:
            content = f.read()

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": content},
            "metadata": {},
        }

    def complete_kernel_type(self, word: str):
        """
        Finds all kernel type that starts like 'word', and
        return the list of all matching kernel types.

        Parameters :
        ---
            - word (str) : the beginning of a kernel type,
                e.g. 'pyth'

        Returns :
        ---
            The list of all matching kernel names, for example
            ['python3', 'python']
        """
        all_matches = []
        self.logger.debug(f"heeere {self.get_available_kernels}")
        for each_kernel in self.get_available_kernels:
            self.logger.debug(f"each kernel {each_kernel}")
            if len(each_kernel) < len(word):
                continue
            potential_match = each_kernel[: len(word)]
            self.logger.debug(
                f"here potential match {potential_match}, len {len(word)}"
            )
            if potential_match == word:
                all_matches.append(each_kernel)
        return all_matches

    def complete_filesystem_path(self, path: str):
        path_obj = Path(path)
        self.logger.debug(str(path_obj))
        if len(path) > 0 and path[-1] == "/":
            parent = path_obj
            name = ""
        else:
            parent = path_obj.parent
            name = path_obj.name

        self.logger.debug(f"last element :{name}")
        self.logger.debug(f"parent : {parent}")

        all_matches = []
        for each_path in parent.iterdir():
            if len(each_path.name) < len(name):
                continue
            potential_match = each_path.name[: len(name)]
            if potential_match == name:
                all_matches.append(str(each_path))
        self.logger.debug(all_matches)
        return all_matches

    def complete_path(self, path: str):
        """
        Completes a path within the kernel tree
        """
        self.logger.debug(f"Completing path {path}")
        path_components = path.split("/")
        last_elem = path_components[-1]
        path_components.pop(-1)

        current_node = self.find_node_by_metadata(self.active_kernel)
        if current_node is None:
            return
        self.logger.debug(
            f"Path components {path_components}, from node {current_node}"
        )
        for component in path_components:
            if component == "..":
                # Move up to the parent node
                current_node = (
                    current_node.parent if current_node.parent else current_node
                )
            else:
                # Find the child node with the corresponding label
                found = False
                for child in current_node.children:
                    if child.value.label == component:
                        current_node = child
                        found = True
                        break

                if not found:
                    return None  # Return None if the path does not exist
        childrens = current_node.children
        all_matches = []
        for each_child in childrens:
            if len(each_child.value.label) < len(last_elem):
                continue
            potential_match = each_child.value.label[: len(last_elem)]
            if potential_match == last_elem:
                all_matches.append("/".join(path_components + [each_child.value.label]))
        return all_matches

    def complete_child_label(self, child_name: str):
        current_node = self.find_node_by_metadata(self.active_kernel)
        if current_node is None:
            return []
        childrens = current_node.children
        all_matches = []
        for each_child in childrens:
            if len(each_child.value.label) < len(child_name):
                continue
            potential_match = each_child.value.label[: len(child_name)]
            if potential_match == child_name:
                all_matches.append(each_child.value.label)
        return all_matches

    def init_commands(self):
        """
        Define all commands of silik programming language.
        """
        cd_parser = KomandParser()
        cd_parser.add_argument("path", label="path", completer=self.complete_path)
        cd_cmd = SilikCommand(self.cd_cmd_handler, cd_parser)

        mkdir_parser = KomandParser()
        mkdir_parser.add_argument(
            "kernel_type", label="kernel_type", completer=self.complete_kernel_type
        )
        mkdir_parser.add_argument("--label", "-l", label="label")
        mkdir_cmd = SilikCommand(self.mkdir_cmd_handler, mkdir_parser)

        ls_parser = KomandParser()
        ls_cmd = SilikCommand(self.ls_cmd_handler, ls_parser)

        restart_parser = KomandParser()
        restart_cmd = SilikCommand(self.restart_cmd_handler, restart_parser)

        run_parser = KomandParser()
        run_parser.add_argument("cmd", label="cmd")
        run_cmd = SilikCommand(self.run_cmd_handler, run_parser)

        kernels_parser = KomandParser()
        kernels_cmd = SilikCommand(self.kernels_cmd_handler, kernels_parser)

        history_parser = KomandParser()
        history_cmd = SilikCommand(self.history_cmd_handler, history_parser)

        branch_parser = KomandParser()
        branch_parser.add_argument(
            "kernel_label", label="kernel_label", completer=self.complete_child_label
        )
        branch_parser.add_argument("--trust", "-t", label="trust_level")
        branch_cmd = SilikCommand(self.branch_cmd_handler, branch_parser)

        detach_parser = KomandParser()
        detach_cmd = SilikCommand(self.detach_cmd_handler, detach_parser)

        help_parser = KomandParser()
        help_cmd = SilikCommand(self.help_cmd_handler, help_parser)

        exit_parser = KomandParser()
        exit_parser.add_argument("--restart", label="restart")
        exit_cmd = SilikCommand(self.exit_cmd_handler, exit_parser)

        source_parser = KomandParser()
        source_parser.add_argument(
            "path", label="path", completer=self.complete_filesystem_path
        )
        source_cmd = SilikCommand(self.source_cmd_handler, source_parser)

        cat_parser = KomandParser()
        cat_parser.add_argument(
            "path", label="path", completer=self.complete_filesystem_path
        )
        cat_cmd = SilikCommand(self.cat_cmd_handler, cat_parser)

        self.all_cmds: dict[str, SilikCommand] = {
            "cd": cd_cmd,
            "mkdir": mkdir_cmd,
            "ls": ls_cmd,
            "tree": ls_cmd,
            "restart": restart_cmd,
            "kernels": kernels_cmd,
            "history": history_cmd,
            "branch": branch_cmd,
            "detach": detach_cmd,
            "run": run_cmd,
            "r": run_cmd,
            "help": help_cmd,
            "exit": exit_cmd,
            "source": source_cmd,
            "cat": cat_cmd,
        }
