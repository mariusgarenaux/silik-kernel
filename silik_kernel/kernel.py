# Basic python dependencies
import os
import shlex
import textwrap
import json
from pathlib import Path
import traceback
from uuid import uuid4, UUID
from typing import Literal, Optional, Tuple, List

# Internal dependencies
from .tools import (
    ALL_KERNELS_LABELS,
    PRETTY_DISPLAY,
    setup_kernel_logger,
    NodeValue,
    TreeNode,
    KernelMetadata,
    KernelFolder,
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
from jupyter_client.manager import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from statikomand import KomandParser

SILIK_VERSION = "1.6.3"


class SilikBaseKernel(Kernel):
    """
    Silik Kernel - Multikernel Manager
    Silik Kernel is MultiKernelManager, wrapped in a jupyter kernel.

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
    code to sub-kernels through appropriate jupyter kernel channels. The commands
    allow to manage kernel with bash-like commands.
    """

    implementation = "Silik"
    implementation_version = SILIK_VERSION
    language = "no-op"
    language_version = SILIK_VERSION
    language_info = {
        "name": "silik",
        "mimetype": "text/plain",
        "file_extension": ".silik",
    }
    banner = "Silik Kernel - Multikernel Manager - Run `help` for commands"
    all_kernels_labels: list[str] = ALL_KERNELS_LABELS
    # help_links: List[dict[str, str]] = [{"text": "", "url": ""}]
    mkm: MultiKernelManager = MultiKernelManager()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_label_rank = 1

        should_custom_log = os.environ.get("SILIK_KERNEL_LOG", "False")
        should_custom_log = (
            True if should_custom_log in ["True", "true", "1"] else False
        )

        if should_custom_log:
            logger = setup_kernel_logger(__name__, self.ident)
            logger.info(f"Started kernel {self.ident} and initalized logger")
            self.logger = logger
        else:
            self.logger = self.log

        self.ksm = KernelSpecManager()
        self.mode: Literal["command", "connect"] = "command"
        self.current_dir: KernelFolder = KernelFolder(label="~", id=str(uuid4()))
        self.tree: TreeNode = TreeNode(self.current_dir)  # the tree is only the
        # root node, but will be updated with other tree nodes :)

        self.kernel_metadata = KernelMetadata(
            id=self.ident, label="home.silik", type="silik", kernel_name="home"
        )
        self.active_kernel: KernelMetadata = self.kernel_metadata
        self.all_kernels: list[KernelMetadata] = []

        self.prettify = True

        self.init_commands()

        self.given_labels = []

    @property
    def active_node(self) -> TreeNode:
        active_node = self.tree.find_node_by_value(self.current_dir)
        if active_node is None:
            raise ValueError(
                "Could not find node for current dir. Please restart silik kernel."
            )
        return active_node

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
            - command mode,
            - connection mode.

        The first one is used to spawn and configure new kernels.
        The second is used to directly connect input of this kernel
        to sub-kernel ones, and to display their output.

        Parameters:
        ---
            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and
                 increases the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after the
                 code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g.
                for Python’s raw_input()).

        Returns:
        ---
            ExecutionResult, according to Jupyter documentation.
        """
        try:
            if code in ["/cmd", "<"]:
                self.mode = "command"
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

            # then either run code, or give it to sub-kernels
            if self.mode == "command":
                self.logger.info("Running in command mode.")
                execution_result, msg = self.do_execute_on_silik(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                if execution_result.get("status") == "error":
                    self.send_response(self.iopub_socket, "error", msg)
                else:
                    self.send_response(self.iopub_socket, "execute_result", msg)
                self.logger.info(f"Output of command : {msg}")
                return execution_result

            elif self.mode == "connect":
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
                self.mode = "command"
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
                    "evalue": str(e),
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
        self.logger.debug(f"Splitted Multiline Code {splitted_code}")

        execution_result, msg = None, None
        for line in splitted_code:
            if line == "":
                continue
            self.logger.info(f"Running line : `{line}`")
            execution_result, msg = self.do_execute_one_command_on_silik(
                line, True, store_history, user_expressions, allow_stdin
            )
            if self.mode == "connect":
                # if mode has switched during execution
                # we stop the cell execution, since the
                # language has changed !
                return execution_result, msg

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
            self.logger.warning(f"Code {code} is empty.")
            return (
                {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                },
                {
                    "ename": "UnknownCommand",
                    "evalue": "Unknown command",
                    "traceback": ["Could not parse command"],
                },
            )
        cmd_name = splitted[0]
        self.logger.debug(f"Splitted command. Command name : `{cmd_name}`.")
        if cmd_name not in self.all_cmds:
            self.logger.warning(f"Command `{cmd_name}` was not found")

            return (
                {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                },
                {
                    "ename": "UnknownCommand",
                    "evalue": cmd_name,
                    "traceback": [
                        f"Command `{cmd_name}` not found. Available commands : {list(self.all_cmds.keys())}"
                    ],
                },
            )
        cmd_obj = self.all_cmds[cmd_name]
        if len(splitted) <= 1:
            self.logger.debug(f"No arguments were found for command {cmd_name}")
            args_str = ""
        else:
            args_str = splitted[1]
            self.logger.debug(f"Arguments for command `{cmd_name}` : `{args_str}`")

        args = cmd_obj.parser.parse_args(shlex.split(args_str))
        self.logger.info(f"Parsed arguments of `{cmd_name}` : `{vars(args)}`")
        try:
            execution_result, msg = cmd_obj.handler(args)
        except Exception as e:
            self.logger.info(f"Error when running command `{cmd_name}` : `{e}`")
            return (
                {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                },
                {
                    "ename": "CommandError",
                    "evalue": str(e),
                    "traceback": traceback.format_exception(e),
                },
            )

        self.logger.debug(f"Output of `{cmd_name}` : `{execution_result}`, `{msg}`")
        return execution_result, msg

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
        self.logger.info(f"Code is sent to selected kernel : {self.current_dir}")

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
        if self.mode == "connect":
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
            self.logger.debug(f"is_complete from {self.active_kernel.label}: {output}")
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
            if self.mode == "connect":
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
            if self.mode == "command":
                ends_with_space = code[-1] == " "
                splitted = code.split(maxsplit=1)
                self.logger.debug(f"Splitted code for completion : {splitted}")
                if len(splitted) == 0:
                    return {
                        "status": "ok",
                        "matches": [],
                        "cursor_start": cursor_pos,
                        "cursor_end": cursor_pos,
                        "metadata": {},
                    }
                if len(splitted) == 1 and not ends_with_space:
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
                    arg_str = splitted[1] if len(splitted) > 1 else " "
                    args = shlex.split(arg_str)
                    if ends_with_space:  # shlex remove space, but we add an empty str
                        # to have completion even for empty strings
                        args += [""]
                    self.logger.debug(f"Completing token list {args}")
                    all_matches = self.all_cmds[cmd_name].parser.complete(args)
                    self.logger.info(f"Completion matches : {all_matches}")

                    return {
                        "status": "ok",
                        "matches": all_matches,
                        "cursor_start": cursor_pos - len(args[-1]),
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
        for each_kernel in self.all_kernels:
            connection_file = each_kernel.connection_file
            if connection_file is not None:
                filename = Path(connection_file).name
                if filename.endswith(".json") and filename.startswith("kernel"):
                    self.logger.info(
                        f"Removing kernel connection file : {connection_file}"
                    )
                    os.remove(connection_file)

        self.mkm.shutdown_all()
        return super().do_shutdown(restart)

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

    def get_kernel_history(
        self, kernel_id: UUID | str, output: bool | None = False
    ) -> list:
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
        if output is None:
            output = False

        if isinstance(kernel_id, UUID):
            kernel_id = str(kernel_id)
        self.logger.debug(f"Getting history for {kernel_id}")
        kc = self.mkm.get_kernel(kernel_id).client()

        try:
            kc.start_channels()

            # Send history request
            msg_id = kc.history(
                raw=True,
                output=output,
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

    def get_kernel_from_label(self, kernel_label: str) -> NodeValue | None:
        """
        Finds a kernel in the tree from its label.
        """
        for each_kernel in self.all_kernels:
            if each_kernel.kernel_name == kernel_label:
                return each_kernel

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

        self.mkm.start_kernel(kernel_name=kernel_name, kernel_id=kernel_id)
        self.given_labels.append(kernel_label)

        km = self.mkm.get_kernel(kernel_id)
        connection_file = os.path.abspath(km.connection_file)
        self.logger.debug(f"Connection file for kernel : {connection_file}")
        kernel_info = self.retrieve_kernel_information(km)
        if kernel_info is not None:
            file_extension = kernel_info.get("language_info", {}).get(
                "file_extension", ""
            )
        else:
            file_extension = ""

        new_kernel = KernelMetadata(
            id=kernel_id,
            label=kernel_label + file_extension,
            kernel_name=kernel_label,
            type=kernel_name,
            kernel_info=kernel_info,
            connection_file=connection_file,
        )
        self.logger.debug(f"Successfully started kernel {new_kernel}")
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
        self.logger.debug(
            f"Created channel with sub-kernel. Sending execute request: {content}."
        )

        msg = kc.session.msg(
            "execute_request",
            content,
        )
        kc.shell_channel.send(msg)
        msg_id = msg["header"]["msg_id"]

        self.logger.debug(f"Sent execute request to kernel. Message id : {msg_id}.")

        while True:
            try:
                msg: JupyterMessage = kc.get_iopub_msg(
                    timeout=60
                )  # pyright: ignore[reportAssignmentType]
            except Exception as e:
                error_msg = (
                    f"IO Pub channel of sub kernel {sub_kernel} timed out (60 sec)."
                )
                self.logger.warning(error_msg)
                kc.stop_channels()
                raise TimeoutError(error_msg)

            self.logger.debug(f"Received msg : {msg}")
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            self.logger.debug(f"Message from sub-kernel : `{msg}`")

            if msg_type in ["execute_result", "display_data"]:
                kc.stop_channels()
                return {
                    "status": "ok",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }, msg
            if msg_type == "stream":
                self.send_response(self.iopub_socket, "stream", msg["content"])

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

    def retrieve_kernel_information(self, kernel_manager: KernelManager) -> dict | None:
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()
        kernel_info = None

        msg_id = kernel_client.kernel_info()
        self.logger.debug(f"Kernel info message id : `{msg_id}`")
        # Send kernel_info_request
        while True:
            msg = kernel_client.get_shell_msg(timeout=5)
            self.logger.debug(f"Message from shell socket : `{msg}`")

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["header"]["msg_type"]

            if msg_type == "kernel_info_reply":
                kernel_info = msg["content"]
                break
            if msg_type == "status":
                if msg["content"]["execution_state"] == "idle":
                    break

        kernel_client.stop_channels()
        self.logger.debug(f"Retrieved kernel informations : `{kernel_info}`")
        return kernel_info

    # ------------------------------------------------------ #
    # ----------------- COMMANDS HANDLERS ------------------ #
    # ------------------------------------------------------ #

    def gateway_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Changes the mode of silik-kernel to 'command'. All future
        code cells will be run on a sub-kernel.

        Positionals arguments :
        ---
            • path (str) : the path (relative or absolute) to the sub-kernel

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py

            In [2]: > k1.py
            Out[2]: All cells are executed on kernel k1.py. Run /cmd to exit this mode and select a new kernel.

            In [3]: 1+1
            Out[1]: 2
        """
        self.mode = "connect"

        node_at_path = self.active_node.find_node_value_from_path(args.path)
        if node_at_path is None:
            raise ValueError(f"Could not find any kernel located at `{args.path}`")

        if not isinstance(node_at_path, KernelMetadata):
            raise ValueError(
                f"The object located at `{args.path}` is not a Kernel but a `{type(node_at_path)}`"
            )

        self.active_kernel = node_at_path

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {
                "text/plain": f"All cells are executed on kernel {self.active_kernel}. Run /cmd to exit this mode and select a new kernel."
            },
            "metadata": {},
        }

    def help_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Display the help message.

        Flags :
        ---
            • cmd (--cmd) : the name of the command

        Example :
        ---
            In [1]: help --cmd kernels
            Out[1]:
            • kernels :
                    Returns the list of available kernel that can be started from silik.

                    Example :
                    ---
                        In [1]: kernels
                        Out[1]: ['python3', 'pydantic_ai', 'octave', 'silik']
        """
        cmd_name = None
        try:
            cmd_name = args.cmd
        except Exception as e:
            pass

        if cmd_name is None:
            content = ""
            for key, value in self.all_cmds.items():
                cmd_help = f"• {key} : "
                cmd_help += f"{value.handler.__doc__}"
                content += cmd_help + "\n"

        else:
            content = f"• {cmd_name} : {self.all_cmds[cmd_name].handler.__doc__}"

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
        """
        Allows to move between the folders (in silik only). Directories can
        be created in silik, but have no link with your real filesystem.

        It is just a way to store and organize the kernel objects.

        Positional arguments :
        ---
            • path (str | None): the path (relative or absolute) towards the new
                directory. If None, go back to home directory (~)

        Example :
        ---
            In [2]: tree
            Out[2]:
            ├─ chatbots
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python <<
               ╰─ k1.py


            In [3]: cd ../chatbots/
            Out[3]: ~/chatbots/

            In [4]: tree
            Out[4]:
            chatbots
            ├─ qwen4b-dist.txt
            ╰─ qwen1b7-local.txt

        """
        found_value = None
        if args.path is None:
            found_value = self.tree.value
        elif isinstance(args.path, list):
            if len(args.path) == 0:
                found_value = self.tree.value
            else:
                found_value = self.active_node.find_node_value_from_path(args.path[0])

        if found_value is None:
            raise ValueError(f"Could not find kernel located at {args.path}")
        if not isinstance(found_value, KernelFolder):
            raise ValueError(
                f"The node located at path {args.path} is not a KernelFolder."
            )
        self.current_dir = found_value
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": self.active_node.path},
            "metadata": {},
        }

    def start_kernel_cmd_handler(
        self, args
    ) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Starts a new kernel, from the root of the selected dir.
        Use tab completion or send 'kernels' command to see the
        list of available kernels.

        Positional arguments :
        ---
            • kernel_type (str) : the type of the kernel which will be started. Must
                be one of available kernels (see `kernels` command)

        Flags :
        ---
            • label (--label, -l) (str) : the label of the started kernel. A random
                label is chosen if not given. The name of the kernel is the label
                followed by the file extension of the kernel.

        Examples :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: start python3 -l k2
            Out[2]:
            ├─ k1.py
            ╰─ k2.py


            In [3]: start bash
            Out[3]:
            ├─ k1.py
            ├─ k2.py
            ╰─ lune.sh

        """

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

        self.active_node.add_children(new_kernel)
        self.all_kernels.append(new_kernel)
        content = self.active_node.path

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

    def tree_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Display the whole tree (directories and kernels) from the current node.

        Example :
        ---
            In [2]: tree
            Out[2]:
            ~
            ├─ chatbots
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python
                ╰─ k1.py
        """
        content = self.active_node.tree_to_str()
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
        """
        Restart a kernel.

        Positional arguments :
        ---
            • path (str) : the path towards the kernel that will be restarted

        Example :
        ---
            In [1]: start python3 -l k1
            Out[1]:
            ╰─ k1.py


            In [2]: run "x=18" k1.py
            Out[2]:

            In [3]: run "x" k1.py
            Out[2]: 18

            In [4]: restart k1.py
            Out[4]: Restarted kernel k1.py

            In [5]: run "x" k1.py
            ---------------------------------------------------------------------------
            NameError                                 Traceback (most recent call last)
            Cell In[1], line 1
            ----> 1 x

            NameError: name 'x' is not defined
        """
        kernel_to_restart = self.active_node.find_node_value_from_path(args.path)
        if not isinstance(kernel_to_restart, KernelMetadata):
            raise ValueError(f"Could not find kernel located at {args.path}")

        self.mkm.restart_kernel(kernel_to_restart.id)
        content = f"Restarted kernel {kernel_to_restart}"
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
        """
        Returns the list of available kernel that can be started from silik.

        Example :
        ---
            In [1]: kernels
            Out[1]: ['python3', 'pydantic_ai', 'octave', 'silik']
        """
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
        """
        Display the history of the selected kernel. Sends an 'history_request' to
        the kernel (see https://jupyter-client.readthedocs.io/en/stable/messaging.html#history).

        > ! not all kernel return information on this message request !

        Positional arguments :
        ---
            • path (str) : the path to the kernel that will send its history

        Optional arguments :
        ---
            • output (-o, --output): a flag, whether or not displaying cells output

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: > k1.py
            Out[2]: All cells are executed on kernel k1.py. Run /cmd to exit this mode and select a new kernel.

            In [3]: x = 19

            In [4]: for i in range(10):
            ...:     print(i*x)
            ...:
            0
            19
            38
            57
            76
            95
            114
            133
            152
            171

            In [5]: /cmd
            Out[5]: Command mode. You can create and select kernels. Send `help` for the list of commands.

            In [6]: history k1.py
            Out[6]:
            ------- 0 -------

            x = 19

            ------- 1 -------

            for i in range(10):
                print(i*x)
        """
        sub_kernel = self.active_node.find_node_value_from_path(args.path)
        if not isinstance(sub_kernel, KernelMetadata):
            raise ValueError(f"Could not find kernel located at {args.path}")

        content = self.get_kernel_history(sub_kernel.id, output=args.output)
        out = ""
        for session, line, code in content:
            indent = 3
            if PRETTY_DISPLAY:

                prefix_in = " " * indent + f"\033[0;32mIn [{line}]:\033[0m"
                prefix_out = " " * indent + f"\033[0;31mOut[{line}]:\033[0m"
            else:
                prefix_in = " " * indent + f"In [{line}]:"
                prefix_out = " " * indent + f"Out[{line}]:"
            preshift = " " * indent + "│" + " " * 3
            end_cell = " " * indent + "╰─" + "─" * 30 + "\n"
            if len(code) == 2:
                input_code, output = code
                out += f"{prefix_in}\n{textwrap.indent(input_code, preshift, predicate=lambda line: True)}\n"
                out += end_cell
                if output is not None:
                    out += f"{prefix_out}\n{textwrap.indent(output, preshift, predicate=lambda line: True)}\n"
                else:
                    out += f"{prefix_out}\n"
                out += end_cell
            else:
                out += f"{prefix_in}\n{textwrap.indent(code, preshift, predicate=lambda line: True)}\n"
                out += end_cell
            out += "\n\n"

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": out},
            "metadata": {},
        }

    def mkdir_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Creates a directory inside the current directory. A directory can be
        used to store kernels. They are not persistent through sessions,
        it is just a way to organize all kernels.

        Positional arguments :
        ---
            • label (str) : the name of the directory to create

        Example :
        ---
            In [1]: mkdir python_kernels
            Out[1]:

            In [2]: cd python_kernels/
            Out[2]: ~/python_kernels/

            In [3]: start python3 -l k1
            Out[3]:
            ╰─ python_kernels <<
                ╰─ k1.py


            In [4]: start python3 -l k2
            Out[4]:
            ╰─ python_kernels <<
                ├─ k1.py
                ╰─ k2.py

        """

        node_value = KernelFolder(label=args.label, id=str(uuid4()))
        self.active_node.add_children(node_value)
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": ""},
            "metadata": {},
        }

    def run_cmd_handler(self, args) -> Tuple[ExecutionResult, IOPubMsg]:
        """
        Send a message to the active sub kernel. Returns the result through
        IOPub channel.

        Positional arguments :
        ---
            • cmd (str) : the command to be sent, between quotes for complex commands
                (e.g. `run "print('hey from ipykernel')" k1.py`)

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: run "x=18" k1.py
            Out[2]:

            In [3]: run "x" k1.py
            Out[2]: 18

        """
        target_kernel = self.active_node.find_node_value_from_path(args.path)

        if not isinstance(target_kernel, KernelMetadata):
            raise ValueError(f"Could not find any kernel located at {args.path}")

        execution_result, jupyter_msg = self.send_code_to_sub_kernel(
            sub_kernel=target_kernel, code=args.cmd, silent=False
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
        """
        Execute the content of a text file on the silik kernel.
        The text file is located on the filesystem where the kernel runs.
        Relative paths are from where you started the jupyter kernel.

        The content must be commands that can be run on silik.
        Multiline commands are supported, but the text file must contain
        only silik commands, not code of an other language.

        Positional arguments :
        ---
            • path (str): the path (relative or absolute) towards
                 the text file

        Example :
        ---
            init.silik :
                ```silik
                start python3 --label k1
                run "x=19" k1.py
                run "x" k1.py
                ```
            In [1]: source init.txt
            Out[1]: 19


        """
        path = args.path

        with open(path, "rt", encoding="utf-8") as f:
            code = f.read()

        return self.do_execute_on_silik(code, False)

    def cat_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Display the content of a text file. The text file is located on the
        filesystem where the kernel runs. Relative paths are from where you started
        the jupyter kernel.

        Positional arguments :
        ---
            • path (str): the path (relative or absolute) towards the text file

        Example :
        ---
            In [3]: cat ex_scripts/ex_init.silik
            Out[3]:
            start python3 -l k1
            run "x=2" k1.py
            run "x" k1.py

        """
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

    def info_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Returns informations about a kernel. Returns the result of a kernel_info_reply,
        see :
        https://jupyter-client.readthedocs.io/en/stable/messaging.html#kernel-info

        Positional arguments :
        ---
            • path (str) : the path to the kernel to which get connection file
                path

        Example :
        ---
            In [1]: start python3 -l k1
            Out[1]:
            ╰─ k1.py

            In [2]: info k1.py
            Out[2]:
            {
                "status": "ok",
                "protocol_version": "5.3",
                "implementation": "ipython",
                "implementation_version": "9.8.0",
                "language_info": {
                    "name": "python",
                    "version": "3.12.12",
                    "mimetype": "text/x-python",
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "pygments_lexer": "ipython3",
                    "nbconvert_exporter": "python",
                    "file_extension": ".py"
                },
                "banner": "Python 3.12.12 (main, Oct 14 2025, 21:38:21) [Clang 20.1.4 ]\nType 'copyright', 'credits' or 'license' for more information\nIPython 9.8.0 -- An enhanced Interactive Python. Type '?' for help.\nTip: Put a ';' at the end of a line to suppress the printing of output.\n",
                "help_links": [...],
                "supported_features": [...]
            }
        """
        kernel = self.active_node.find_node_value_from_path(args.path)
        if not isinstance(kernel, KernelMetadata):
            raise ValueError(f"Could not find kernel located at {args.path}")
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": json.dumps(kernel.kernel_info, indent=4)},
            "metadata": {},
        }

    def connection_file_cmd_handler(
        self, args
    ) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Returns the path to the connection file kernel.

        Positional arguments :
        ---
            • path (str) : the path to the kernel to which get connection file
                path


        Example :
        ---
            In [1]: start python3 -l k1
            Out[1]:
            ╰─ k1.py

            In [2]: connection_file k1.py
            Out[2]: /Users/mgg/silik-kernel/kernel-86a19659-4598-4c82-9bc2-c2595310cf2c.json
        """
        kernel = self.active_node.find_node_value_from_path(args.path)
        if not isinstance(kernel, KernelMetadata):
            raise ValueError(f"Could not find kernel located at {args.path}")

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": kernel.connection_file},
            "metadata": {},
        }

    def ls_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Displays the content of a folder (in silik kernel).

        Positional arguments :
        ---
            • path (str | None): the path to the directory. If None,
                displays the content of the current dir.

        Example :
        ---
            In [11]: tree
            Out[11]:
            ~
            ├─ chatbots
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python
               ╰─ k1.py


            In [12]: cd chatbots/
            Out[12]: ~/chatbots/

            In [13]: ls
            Out[13]:
            qwen4b-dist.txt
            qwen1b7-local.txt
        """

        target_node = None
        if args.path is None:
            target_node = self.active_node
        elif isinstance(args.path, list):
            if len(args.path) == 0:
                target_node = self.active_node
            else:
                target_node = self.active_node.find_node_from_path(args.path[0])

        if target_node is None:
            raise ValueError(f"Could not find any folder at {args.path}")

        if target_node.node_type != "folder":
            raise ValueError(
                f"`ls` command only works for directory, not `{type(target_node)}`"
            )

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": target_node.childrens_to_str()},
            "metadata": {},
        }

    def pwd_cmd_handler(self, args) -> Tuple[ExecutionResult, ExecuteResultContent]:
        """
        Prints the current working directory (path from ~).

        Example :
        ---
            In [8]: tree
            Out[8]:
            chatbots
            ├─ qwen4b-dist.txt
            ╰─ qwen1b7-local.txt


            In [9]: pwd
            Out[9]: ~/chatbots/
        """
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, {
            "execution_count": self.execution_count,
            "data": {"text/plain": self.active_node.path},
            "metadata": {},
        }

    def complete_help_cmd(self, word, rank):
        return self.complete_cmd_name(word)

    def complete_kernels_cmd(self, word, rank):
        self.logger.debug(f"Completing kernels : {word}, {rank}")
        return self.complete_kernel_type(word)

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
        for each_kernel in self.get_available_kernels:
            if len(each_kernel) < len(word):
                continue
            potential_match = each_kernel[: len(word)]
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
                if each_path.is_dir():
                    all_matches.append(str(each_path) + "/")
                else:
                    all_matches.append(str(each_path))

        self.logger.debug(all_matches)
        return all_matches

    def complete_cmd_name(self, cmd_name):
        self.logger.debug(f"Completing cmd name {cmd_name}")
        all_matches = []
        for each_cmd in self.all_cmds:
            if len(each_cmd) < len(cmd_name):
                continue
            potential_match = each_cmd[: len(cmd_name)]
            if potential_match == cmd_name:
                all_matches.append(each_cmd)
        return all_matches

    def complete_path(self, path: str, keep_files: bool = True) -> list[str]:
        """
        Completes a path within the kernel tree

        Parameters :
        ---
            - path (str): the path that need to be completed
            - filter_dir (bool = False) : whether or not returning
                matches among file names. If False, only matching
                directories are returned. Else, the file with matching
                names are also returned.
        """
        self.logger.debug(f"Completing path {path}")
        path_components = path.split("/")
        last_elem = path_components[-1]
        path_components.pop(-1)

        current_node = self.active_node
        if current_node is None:
            return []
        self.logger.debug(
            f"Path components {path_components}, from node {current_node}"
        )
        parent_last_elem = self.active_node.find_node_from_path(
            "/".join(path_components)
        )

        all_matches = []
        if parent_last_elem is None:
            return all_matches

        for each_child in parent_last_elem.children:
            if len(each_child.value.label) < len(last_elem):
                continue
            potential_match = each_child.value.label[: len(last_elem)]
            if potential_match == last_elem:
                if each_child.node_type == "leaf" and keep_files:
                    all_matches.append(
                        "/".join(path_components + [each_child.value.label])
                    )
                elif each_child.node_type == "folder":
                    all_matches.append(
                        "/".join(path_components + [each_child.value.label]) + "/"
                    )
        return all_matches

    def complete_child_label(self, child_name: str):
        current_node = self.active_node
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

    def complete_source_cmd(self, word: str, rank: int | None):
        matches = self.complete_filesystem_path(word)
        self.logger.debug(f"Completing path : {word}")
        if matches is None:
            return [word]
        else:
            return matches

    def complete_local_dir_arg(self, word: str, rank: int | None) -> list[str]:
        """
        Completes the path only to directories
        """
        matches = self.complete_path(word, keep_files=False)
        if matches is None:
            return [word]
        else:
            return matches

    def complete_local_path_arg(self, word: str, rank: int | None) -> list[str]:
        """
        Complete the path to either a directory or a kernel
        """
        matches = self.complete_path(word)
        self.logger.debug(f"Matches from local path : {matches}")
        if matches is None:
            return [word]
        else:
            return matches

    def complete_filesystem_path_arg(self, word, rank: int | None) -> list[str]:
        matches = self.complete_filesystem_path(word)
        if matches is None:
            return [word]
        else:
            return matches

    def init_commands(self):
        """
        Define all commands of silik programming language.
        """
        cd_parser = KomandParser("cd")
        cd_parser.add_argument("path", nargs="*", completer=self.complete_local_dir_arg)
        cd_cmd = SilikCommand(self.cd_cmd_handler, cd_parser)

        mkdir_parser = KomandParser("mkdir")
        mkdir_parser.add_argument("label")
        mkdir_cmd = SilikCommand(self.mkdir_cmd_handler, mkdir_parser)

        tree_parser = KomandParser("ls")
        tree_cmd = SilikCommand(self.tree_cmd_handler, tree_parser)

        restart_parser = KomandParser("restart")
        restart_parser.add_argument("path", completer=self.complete_local_path_arg)
        restart_cmd = SilikCommand(self.restart_cmd_handler, restart_parser)

        run_parser = KomandParser("run")
        run_parser.add_argument("cmd")
        run_parser.add_argument("path", completer=self.complete_local_path_arg)
        run_cmd = SilikCommand(self.run_cmd_handler, run_parser)

        kernels_parser = KomandParser("kernels")
        kernels_cmd = SilikCommand(self.kernels_cmd_handler, kernels_parser)

        info_parser = KomandParser("info")
        info_parser.add_argument("path", completer=self.complete_local_path_arg)
        info_cmd = SilikCommand(self.info_cmd_handler, info_parser)

        history_parser = KomandParser("history")
        history_parser.add_argument("path", completer=self.complete_local_path_arg)
        history_parser.add_argument(
            "--output", "-o", dest="output", action="store_true"
        )
        history_cmd = SilikCommand(self.history_cmd_handler, history_parser)

        help_parser = KomandParser("help")
        help_parser.add_argument("--cmd", completer=self.complete_help_cmd)
        help_cmd = SilikCommand(self.help_cmd_handler, help_parser)

        start_kernel_parser = KomandParser("start")
        start_kernel_parser.add_argument(
            "kernel_type", completer=self.complete_kernels_cmd
        )
        start_kernel_parser.add_argument("--label", "-l")
        start_kernel_cmd = SilikCommand(
            self.start_kernel_cmd_handler, start_kernel_parser
        )

        ls_parser = KomandParser("ls")
        ls_parser.add_argument(
            "path", nargs="*", completer=self.complete_local_path_arg
        )
        ls_cmd = SilikCommand(self.ls_cmd_handler, ls_parser)

        source_parser = KomandParser("source")
        source_parser.add_argument("path", completer=self.complete_filesystem_path_arg)
        source_cmd = SilikCommand(self.source_cmd_handler, source_parser)

        cat_parser = KomandParser("cat")
        cat_parser.add_argument("path", completer=self.complete_filesystem_path_arg)
        cat_cmd = SilikCommand(self.cat_cmd_handler, cat_parser)

        gateway_parser = KomandParser("gateway")
        gateway_parser.add_argument("path", completer=self.complete_local_path_arg)
        gateway_cmd = SilikCommand(self.gateway_cmd_handler, gateway_parser)

        pwd_parser = KomandParser("pwd")
        pwd_cmd = SilikCommand(self.pwd_cmd_handler, pwd_parser)

        connection_file_parser = KomandParser("connection_file")
        connection_file_parser.add_argument(
            "path", completer=self.complete_local_path_arg
        )
        connection_file_cmd = SilikCommand(
            self.connection_file_cmd_handler, connection_file_parser
        )

        self.all_cmds: dict[str, SilikCommand] = {
            "kernels": kernels_cmd,
            "start": start_kernel_cmd,
            "restart": restart_cmd,
            "info": info_cmd,
            "connection_file": connection_file_cmd,
            "history": history_cmd,
            "run": run_cmd,
            "source": source_cmd,
            "cat": cat_cmd,
            ">": gateway_cmd,
            "gateway": gateway_cmd,
            "tree": tree_cmd,
            "mkdir": mkdir_cmd,
            "cd": cd_cmd,
            "ls": ls_cmd,
            "pwd": pwd_cmd,
            "help": help_cmd,
        }
