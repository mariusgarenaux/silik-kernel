# Basic python dependencies
import os
import shlex
import textwrap
import json
from pathlib import Path
import traceback
from uuid import uuid4
import asyncio
import glob
from typing import Literal, Optional

# Internal dependencies
from .tools import (
    ALL_KERNELS_LABELS,
    PRETTY_DISPLAY,
    add_custom_logger_handler,
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
)

# External dependencies
from ipykernel.kernelbase import Kernel
from jupyter_client.multikernelmanager import (
    AsyncMultiKernelManager,
)
from jupyter_client.manager import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_client.connect import find_connection_file
from ipykernel.connect import get_connection_file

from statikomand import KomandParser

SILIK_VERSION = "1.6.6"


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
    mkm: AsyncMultiKernelManager = AsyncMultiKernelManager()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_label_rank = 1

        if os.environ.get("SILIK_KERNEL_LOG_LEVEL", False):
            add_custom_logger_handler(self.log)

        self.log.info(f"Started kernel {self.ident} and initalized logger")

        self.ksm: KernelSpecManager = KernelSpecManager()
        self.log.info(self.ksm.find_kernel_specs())
        self.mode: Literal["command", "connect"] = "command"
        self.current_dir: KernelFolder = KernelFolder(label="~", id=str(uuid4()))
        self.tree: TreeNode = TreeNode(self.current_dir)  # the tree is only the
        # root node, but will be updated with other tree nodes :-)

        self.kernel_metadata = KernelMetadata(
            id=self.ident,
            label="home.silik",
            type="silik",
            kernel_name="home",
            remote_connection_file=False,
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

    async def do_execute(  # pyright: ignore[reportIncompatibleMethodOverride]
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
            self.payload = []
            self.kernel_resp: ExecutionResult = {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }
            if code in ["exit", "exit()", "quit", "quit()"]:
                # return self.do_shutdown(False)
                self.payload = [{"source": "ask_exit", "keepkernel": False}]
                self.kernel_resp["payload"] = self.payload
                return self.kernel_resp
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
                return self.kernel_resp

            # then either run code, or give it to sub-kernels
            if self.mode == "command":
                self.log.info("Running in command mode.")
                await self.do_execute_on_silik(
                    code, silent, store_history, user_expressions, allow_stdin
                )
            elif self.mode == "connect":
                await self.do_execute_on_sub_kernel(
                    self.active_kernel,
                    code,
                    silent,
                    store_history,
                    user_expressions,
                    allow_stdin,
                )
            else:
                self.mode = "command"
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
            self.kernel_resp["status"] = "error"
            return self.kernel_resp
        else:
            if "payload" in self.kernel_resp:
                self.kernel_resp["payload"] = self.payload
            return self.kernel_resp

    async def do_execute_on_silik(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> Optional[IOPubMsg]:
        """
        Do execute method for "command mode". Runs command on silik kernel.
        Commands can be multiline commands.

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
        self.log.debug(f"Splitted Multiline Code {splitted_code}")

        for line in splitted_code:
            if line == "":
                continue
            self.log.info(f"Running line : `{line}`")
            await self.do_execute_one_command_on_silik(
                line, True, store_history, user_expressions, allow_stdin
            )
            self.send_stream("\n")
            if self.mode == "connect":
                # if mode has switched during execution
                # we stop the cell execution, since the
                # language has changed !
                return

    async def do_execute_one_command_on_silik(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
        cmd_stdin: str | None = None,
    ) -> Optional[IOPubMsg]:
        """
        Execute one of command in silik. Do not send result to iopub socket;
        but returns the message that is meant to be send to iopub socket,
        as well as the execution result.

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
            self.log.info(f"Code {code} is empty.")
            self.kernel_resp["status"] = "error"

            return {
                "ename": "UnknownCommand",
                "evalue": "Unknown command",
                "traceback": ["Could not parse command"],
            }
        cmd_name = splitted[0]
        self.log.debug(f"Splitted command. Command name : `{cmd_name}`.")
        if cmd_name not in self.all_cmds:
            self.log.info(f"Command `{cmd_name}` was not found")
            self.kernel_resp["status"] = "error"
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "UnknownCommand",
                    "evalue": cmd_name,
                    "traceback": [
                        f"Command `{cmd_name}` not found. Available commands : {list(self.all_cmds.keys())}"
                    ],
                },
            )
            return
        cmd_obj = self.all_cmds[cmd_name]
        if len(splitted) <= 1:
            self.log.debug(f"No arguments were found for command {cmd_name}")
            args_str = ""
        else:
            args_str = splitted[1]
            self.log.debug(f"Arguments for command `{cmd_name}` : `{args_str}`")
        a = shlex.split(args_str)
        self.log.debug(f"Trying to parse arguments with parser : {a}")
        try:
            if "-h" in a or "--help" in a:

                class FakeNameSpace:
                    def __init__(self, cmd_name):
                        self.cmd = cmd_name

                msg = self.help_cmd_handler(FakeNameSpace(cmd_name))
            else:
                args = cmd_obj.parser.parse_args(a)
                self.log.info(f"Parsed arguments of `{cmd_name}` : `{vars(args)}`")
                msg = await cmd_obj.run(args)
            return msg
        except Exception as e:
            self.log.info(f"Error when running command `{cmd_name}` : `{e}`")
            self.kernel_resp["status"] = "error"
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "CommandError",
                    "evalue": str(e),
                    "traceback": traceback.format_exception(e),
                },
            )

    async def do_execute_on_sub_kernel(
        self,
        sub_kernel: KernelMetadata,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ):
        """
        Send code to sub kernel for execution through shell channel. Opens a client with
        the kernel, by using MultiKernelManager from JupyterClient.
        We then we listen on:
            - IOPub Channel for execution result
            - Shell Channel for optional payloads that could be forwarded to frontend
            - Stdin Channel for kernel asking input_request to frontend

        The kernel response (for silik kernel) is filled with appropriate informations,
        and messages from each of the above sockets are forwarded to silik kernel frontend.

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
            The JupyterMessage from sub-kernel.
        """
        kc = self.create_kernel_client(sub_kernel)
        kc.start_channels()
        content: ExecuteRequestContent = {
            "code": code,
            "silent": silent,
            "store_history": store_history,
            "user_expressions": user_expressions,
            "allow_stdin": allow_stdin,
            "stop_on_error": True,
        }
        self.log.debug(
            f"Created channel with sub-kernel. Sending execute request: {content}."
        )

        msg_sent = kc.session.msg("execute_request", content)
        kc.shell_channel.send(msg_sent)
        msg_id = msg_sent["header"]["msg_id"]

        self.log.debug(f"Sent execute request to kernel. Message id : {msg_id}.")
        self.forward_sub_kernel_channels = True
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.forward_sub_kernel_iopub_channel(kc, msg_id))
                tg.create_task(
                    self.forward_sub_kernel_stdin_channel(kc, msg_id, allow_stdin)
                )
                tg.create_task(self.forward_sub_kernel_shell_channel(kc, msg_id))
        except* SubKernelExecutionDone:
            self.log.info("Execution on sub kernel ended")
        kc.stop_channels()

    async def do_is_complete(self, code: str):  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.mode == "connect":
            kc = self.create_kernel_client(self.active_kernel)
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
                msg = await kc.get_shell_msg()
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]

                if msg_type == "is_complete_reply":
                    output = msg["content"]
                    break

                elif msg_type == "error":
                    output = msg["content"]
                    break
            self.log.debug(f"is_complete from {self.active_kernel.label}: {output}")
            kc.stop_channels()
            if len(output) == 0:
                return {"status": "unknown"}
            return output
        if code.endswith(" "):
            return {"status": "incomplete", "indent": ""}
        return {"status": "unknown"}

    async def do_complete(self, code: str, cursor_pos: int):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Tab completion. Two modes :
            - connect : gateway towards tab completion of sub kernel
            - command : complete command names from self.all_cmds, also completes
                the commands argument with any function declared as a completer
        """
        try:
            if self.mode == "connect":
                # just act as a gateway towards active kernel
                kc = self.create_kernel_client(self.active_kernel)
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
                    msg = await kc.get_shell_msg()
                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue

                    msg_type = msg["msg_type"]

                    if msg_type == "complete_reply":
                        output = msg["content"]
                        break

                    elif msg_type == "error":
                        output = msg["content"]
                        break
                self.log.debug(f"do complete from {self.active_kernel.label}: {output}")
                kc.stop_channels()
                if len(output) > 0:
                    return output
            if self.mode == "command":
                ends_with_space = code[-1] == " "
                splitted = code.split(maxsplit=1)
                self.log.debug(f"Splitted code for completion : {splitted}")
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
                    self.log.debug(f"Completing token list {args}")
                    all_matches = self.all_cmds[cmd_name].parser.complete(args)
                    self.log.info(f"Completion matches : {all_matches}")

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
            self.log.info(traceback.format_exception(e))
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

    async def interrupt_request(self, stream, ident, parent):
        """
        This method is called when an interrupt_request message is received.

        By default, Ipykernel restart the kernel on an interrupt request.
        We override this here, to either :
            - forward to the active sub kernel an interrupt message (in the
            case it accepts them)
            - stop the current cell from running (if in command mode),
            and make custom housekeeping
        """
        self.log.info("Cell interuption asked ")
        if not self.session:
            return

        if self.mode == "command":
            content = {"status": "ok"}
            self.session.send(stream, "interrupt_reply", content, parent, ident=ident)  # pyright: ignore[reportAttributeAccessIssue]
            return

        if self.mode == "connect":
            # km: AsyncKernelManager = self.mkm.get_kernel(self.active_kernel.id)  # pyright: ignore[reportAssignmentType]
            # self.log.debug(f"Sending do_complete to {self.active_kernel}")
            # if km.kernel_spec is None:
            #     self.log.warning(
            #         f"Kernelspec not known of kernel `{self.active_kernel}`. Can't send an `interrupt message`."
            #     )
            #     return
            # if km.kernel_spec.interrupt_mode != "message":
            #     self.log.info(
            #         f"Interrupt messages are transferred only when the sub kernel has `interrupt_mode` set to `message`. See https://jupyter-client.readthedocs.io/en/stable/kernels.html. This is not the case for kernel `{self.active_kernel}`."
            #     )
            #     return

            kc = self.create_kernel_client(self.active_kernel)

            kc.start_channels()
            msg = kc.session.msg("interrupt_request", {})
            kc.control_channel.send(msg)
            msg_id = msg["header"]["msg_id"]
            output = {}

            while True:
                msg = await kc.get_control_msg()
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]

                if msg_type == "interrupt_reply":
                    content = msg["content"]
                    self.log.info(f"Content of interrupt reply : {content}")
                    # /!\ for ipykernel subclasses, if the interrupt_request is not
                    # overridden, the kernel restarts on cell interruption.

                    self.session.send(  # pyright: ignore[reportAttributeAccessIssue]
                        stream, "interrupt_reply", content, parent, ident=ident
                    )
                    kc.stop_channels()
                    return

                elif msg_type == "error":
                    output = msg["content"]
                    break

            self.log.debug(f"interrupt_reply from {self.active_kernel.label}: {output}")
            kc.stop_channels()

    # TODO: implement subshell logic to be able to run a kernel directly from subshell
    # -> this could allow asynchronous multi code settings  (in command mode) ?
    # -> and in connect mode, we could act as a gateway with subshells of sub-kernels ?
    # async def list_subshell_request(self, socket, ident, parent) -> None:
    #     """
    #     Returns the list of all kernels managed by this instance of silik.

    #     Message forward with subshell id in shell message headers is not
    #     yet implemented, but should soon.
    #     Useless until the jupyter frontends implements the subshell
    #     features.
    #     """
    #     reply = [each_kernel.id for each_kernel in self.all_kernels]
    #     self.session.send(socket, "list_subshell_reply", reply, parent, ident)

    async def do_shutdown(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, restart: bool
    ) -> dict:
        """
        Shutdown all sub kernels, and remove their connection files.
        """

        # connection file should be removed by sub kernel themselves
        # in case, we double check
        for each_kernel in self.all_kernels:
            if (
                each_kernel.remote_connection_file
            ):  # don't remove remote connection files
                continue
            connection_file = each_kernel.connection_file
            if connection_file is not None:
                filename = Path(connection_file).name
                if filename.endswith(".json") and filename.startswith("kernel"):
                    if os.path.isfile(connection_file):
                        os.remove(connection_file)

        await self.mkm.shutdown_all()

        return super().do_shutdown(restart)

    # ------------------------------------------------------ #
    # ------------- tools for executing code --------------- #
    # ------------------------------------------------------ #

    def create_kernel_client(self, kernel: KernelMetadata) -> AsyncKernelClient:
        """
        Either uses kernel manager to create a kernel client (for kernel
        started by this process); or creates an AsyncKernelClient with
        an exisiting kernel connection file.
        """
        if kernel.remote_connection_file:
            kc: AsyncKernelClient = AsyncKernelClient(
                connection_file=kernel.connection_file
            )
            kc.load_connection_file()

            return kc
        km: AsyncKernelManager = self.mkm.get_kernel(kernel.id)  # pyright: ignore[reportAssignmentType]
        return km.client()

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

    async def get_kernel_history(
        self, kernel: KernelMetadata, output: bool | None = False
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

        kc = self.create_kernel_client(kernel)

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
                msg = await kc.get_shell_msg()
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                if msg["msg_type"] == "history_reply":
                    self.log.debug(f"Kernel history : {msg['content']}")
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

    async def connect_to_living_kernel(
        self, connection_file_path: str, kernel_label: str | None = None
    ) -> KernelMetadata:
        """
        Creates a KernelMetadata from a living kernel.
        """
        kernel_label = self.new_kernel_label(kernel_label)

        kc: AsyncKernelClient = AsyncKernelClient(connection_file=connection_file_path)
        kc.load_connection_file()
        self.log.info(f"Kernel client : {kc}")
        kernel_info = await self.retrieve_kernel_information(kc)
        if kernel_info is None:
            raise ValueError(
                f"Could not retrieve kernel information from `{connection_file_path}`"
            )

        file_extension = kernel_info.get("language_info", {}).get("file_extension", "")

        return KernelMetadata(
            id=str(uuid4()),
            label=kernel_label + file_extension,
            kernel_name=kernel_label,
            type=kernel_info["language_info"]["name"],
            kernel_info=kernel_info,
            connection_file=connection_file_path,
            remote_connection_file=True,
        )

    def new_kernel_label(self, kernel_label):
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
            self.log.debug("Existing label")
            raise ValueError(
                f"Existing label for kernel : `{kernel_label}`. Choose an other one."
            )
        return kernel_label

    async def start_kernel(
        self, kernel_name: str, kernel_label: str | None = None
    ) -> KernelMetadata:
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
        self.log.debug(f"Starting new kernel of type : {kernel_name}")
        kernel_id = str(uuid4())
        kernel_label = self.new_kernel_label(kernel_label)

        await self.mkm.start_kernel(kernel_name=kernel_name, kernel_id=kernel_id)
        self.given_labels.append(kernel_label)

        km: AsyncKernelManager = self.mkm.get_kernel(kernel_id)  # pyright: ignore[reportAssignmentType]
        connection_file = os.path.abspath(km.connection_file)
        self.log.debug(f"Connection file for kernel : {connection_file}")

        kc = km.client()
        kernel_info = await self.retrieve_kernel_information(kc)
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
            remote_connection_file=False,
        )
        self.log.debug(f"Successfully started kernel {new_kernel}")
        return new_kernel

    async def retrieve_kernel_information(
        self, kernel_client: AsyncKernelClient
    ) -> dict | None:

        kernel_client.start_channels()
        kernel_info = None

        msg_id = kernel_client.kernel_info()
        self.log.debug(f"Kernel info message id : `{msg_id}`")
        # Send kernel_info_request
        while True:
            msg = await kernel_client.get_shell_msg(timeout=5)
            self.log.debug(f"Message from shell socket : `{msg}`")

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
        self.log.debug(f"Retrieved kernel informations : `{kernel_info}`")
        return kernel_info

    async def forward_sub_kernel_stdin_channel(
        self, kc: AsyncKernelClient, msg_id: str, allow_stdin: bool
    ):
        """
        Listen on sub kernel stdin channel, in case an input_request
        message is sent by sub kernel.
        In this case, the silik kernel sends the very same input_request
        message to the frontend.
        """
        while self.forward_sub_kernel_channels:
            if not allow_stdin:
                # no need to listen on stdin
                break
            try:
                stdin_msg = await kc.get_stdin_msg(timeout=1)
            except Exception as e:
                self.log.info(
                    f"Error while listening on sub kernel stdin channel : {e}"
                )
                continue

            else:
                if stdin_msg["parent_header"].get("msg_id") != msg_id:
                    continue
                self.log.debug(f"Message from stdin : {stdin_msg}.")
                self._allow_stdin = True
                out = self.raw_input(stdin_msg["content"]["prompt"])
                self._allow_stdin = False
                self.log.debug(f"out of stdin : {out}")
                input_reply = kc.session.msg("input_reply", {"value": out})
                kc.stdin_channel.send(input_reply)

    async def forward_sub_kernel_shell_channel(
        self, kc: AsyncKernelClient, msg_id: str
    ):
        """
        Listen on shell channel, until a message with the matching id
        is found.
        Then, fills self.payload with the payload from the shell message,
        and breaks the loop.
        """
        while self.forward_sub_kernel_channels:
            try:
                shell_msg: JupyterMessage = await kc.get_shell_msg(timeout=5)  # pyright: ignore[reportAssignmentType]
            except Exception as e:
                self.log.info(
                    f"Error while listening on sub kernel shell channel : {e}"
                )
                continue
            else:
                if shell_msg["parent_header"].get("msg_id") != msg_id:
                    continue

                if shell_msg["msg_type"] == "execute_reply":
                    new_payload = shell_msg["content"]["payload"]
                    if len(new_payload) > 0:
                        self.payload += shell_msg["content"]["payload"]

    async def forward_sub_kernel_iopub_channel(
        self, kc: AsyncKernelClient, msg_id: str
    ):
        """
        Listen on IOPub channel of subkernel, and forwards message to silik kernel frontend.
        Change the self.kernel_resp on the fly.
        """
        while True:
            try:
                jupyter_msg: JupyterMessage = await kc.get_iopub_msg(timeout=5)  # pyright: ignore[reportAssignmentType]
            except Exception as e:
                self.log.info(f"Error when getting iopub message : {e}")
                continue
            else:
                self.log.debug(f"Received msg : {jupyter_msg}")
                if jupyter_msg["parent_header"].get("msg_id") != msg_id:
                    continue

                iopub_msg_type = jupyter_msg["msg_type"]
                self.log.info(f"Message from sub-kernel : `{jupyter_msg}`")

                if iopub_msg_type in ["execute_result", "display_data"]:
                    self.send_response(
                        self.iopub_socket, iopub_msg_type, jupyter_msg["content"]
                    )
                    break

                if iopub_msg_type == "stream":
                    self.send_response(
                        self.iopub_socket, "stream", jupyter_msg["content"]
                    )

                if iopub_msg_type == "error":
                    self.kernel_resp["status"] = "error"
                    self.send_response(
                        self.iopub_socket, "error", jupyter_msg["content"]
                    )
                    break

                if (
                    iopub_msg_type == "status"
                    and jupyter_msg["content"]["execution_state"] == "idle"
                ):
                    self.send_response(
                        self.iopub_socket, "status", jupyter_msg["content"]
                    )
                    break
        self.forward_sub_kernel_channels = False
        raise SubKernelExecutionDone
        # stop listening on sub kernel channels
        # when the code was executed

    # ------------------------------------------------------ #
    # ----------------- COMMANDS HANDLERS ------------------ #
    # ------------------------------------------------------ #

    def gateway_cmd_handler(self, args):
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

        node_at_path = self.active_node.find_node_value_from_path(args.path)
        if node_at_path is None:
            raise ValueError(f"Could not find any kernel located at `{args.path}`")

        if not isinstance(node_at_path, KernelMetadata):
            raise ValueError(
                f"The object located at `{args.path}` is not a Kernel but a `{type(node_at_path)}`"
            )

        self.active_kernel = node_at_path

        self.mode = "connect"
        self.send_stream(
            f"All cells are executed on kernel {self.active_kernel}. Run /cmd to exit this mode and select a new kernel."
        )

    def help_cmd_handler(self, args):
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
        except Exception:
            pass

        if cmd_name is None:
            content = ""
            for key, value in self.all_cmds.items():
                cmd_help = f"• {key} : "
                cmd_help += f"{value.handler.__doc__}"
                content += cmd_help + "\n"

        else:
            content = f"• {cmd_name} : {self.all_cmds[cmd_name].handler.__doc__}"

        self.send_stream(content)

    def cd_cmd_handler(self, args):
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
        self.send_stream(self.active_node.path)

    async def new_kernel_cmd_handler(self, args):
        """
        Opens a new kernel, from the root of the selected dir.
        If a kernel name is given (python3, ...), the kernel is
        started as a subprocess of this one.
        You can also give the path to a connection file, and silik
        kernel will connect directly with this kernel through the
        connection file.
        Use tab completion or send 'kernels' command to see the
        list of available kernels.

        Positional arguments :
        ---
            • kernel (str) : the type of the kernel which will be started, or
            the path towards a connection file. See `kernels` command)

        Flags :
        ---
            • label (--label, -l) (str) : the label of the started kernel. A random
                label is chosen if not given. The name of the kernel is the label
                followed by the file extension of the kernel.

        Examples :
        ---
            In [1]: new python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: new python3 -l k2
            Out[2]:
            ├─ k1.py
            ╰─ k2.py


            In [3]: new bash
            Out[3]:
            ├─ k1.py
            ├─ k2.py
            ╰─ lune.sh

        """

        if not args.kernel:
            raise ValueError(
                f"Please specify a kernel among {self.get_available_kernels}; or a path towards an existing connection file."
            )

        if args.kernel in self.get_available_kernels:
            new_kernel = await self.start_kernel(
                args.kernel, None if not args.label else args.label
            )
            if new_kernel is None:
                raise ValueError(
                    f"The label {args.label} already exists, please choose an other one."
                )
        elif os.path.isfile(args.kernel):
            new_kernel = await self.connect_to_living_kernel(
                args.kernel, None if not args.label else args.label
            )
        else:
            raise ValueError(
                "The given parameter is neither a kernel type, nor a path towards a connection file."
            )

        self.active_node.add_children(new_kernel)
        self.all_kernels.append(new_kernel)
        content = self.active_node.path

        self.send_stream(content)

    def tree_cmd_handler(self, args):
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
        self.send_stream(content)

    async def restart_cmd_handler(self, args):
        """
        Restart a kernel.

        Positional arguments :
        ---
            • path (str) : the path towards the kernel that will be restarted

        Example :
        ---
            In [1]: new python3 -l k1
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

        await self.mkm.restart_kernel(kernel_to_restart.id)
        content = f"Restarted kernel {kernel_to_restart}"
        self.send_stream(content)

    def kernels_cmd_handler(self, args):
        """
        Returns the list of available kernel that can be started from silik.

        Example :
        ---
            In [1]: kernels
            Out[1]: ['python3', 'pydantic_ai', 'octave', 'silik']
        """
        content = self.get_available_kernels
        self.send_stream(content)

    async def history_cmd_handler(self, args):
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
            In [1]: new python3 --label k1
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

        content = await self.get_kernel_history(sub_kernel, output=args.output)
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
        self.send_stream(out)

    def mkdir_cmd_handler(self, args):
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

            In [3]: new python3 -l k1
            Out[3]:
            ╰─ python_kernels <<
                ╰─ k1.py


            In [4]: new python3 -l k2
            Out[4]:
            ╰─ python_kernels <<
                ├─ k1.py
                ╰─ k2.py

        """

        node_value = KernelFolder(label=args.label, id=str(uuid4()))
        self.active_node.add_children(node_value)

    async def run_cmd_handler(self, args):
        """
        Send a message to the active sub kernel. Returns the result in an
        IOPubMsg.

        Positional arguments :
        ---
            • cmd (str) : the command to be sent, between quotes for complex commands
                (e.g. `run "print('hey from ipykernel')" k1.py`)

        Example :
        ---
            In [1]: new python3 --label k1
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
        await self.do_execute_on_sub_kernel(
            sub_kernel=target_kernel, code=args.cmd, silent=False
        )

    def exit_cmd_handler(self, args):
        raise NotImplementedError("Exit command is not yet implemented.")

    async def source_cmd_handler(self, args):
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
                new python3 --label k1
                run "x=19" k1.py
                run "x" k1.py
                ```
            In [1]: source init.txt
            Out[1]: 19


        """
        path = args.path

        with open(path, "rt", encoding="utf-8") as f:
            code = f.read()

        await self.do_execute_on_silik(code, False)

    def cat_cmd_handler(self, args):
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
            new python3 -l k1
            run "x=2" k1.py
            run "x" k1.py

        """
        path = args.path
        with open(path, "rt", encoding="utf-8") as f:
            content = f.read()

        self.send_stream(content)

    def info_cmd_handler(self, args):
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
            In [1]: new python3 -l k1
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
        content = json.dumps(kernel.kernel_info, indent=4)
        self.send_stream(content)

    def connection_file_cmd_handler(self, args):
        """
        Returns the path to the connection file kernel.

        Positional arguments :
        ---
            • path (str) : the path to the kernel to which get connection file
                path


        Example :
        ---
            In [1]: new python3 -l k1
            Out[1]:
            ╰─ k1.py

            In [2]: connection_file k1.py
            Out[2]: /Users/mgg/silik-kernel/kernel-86a19659-4598-4c82-9bc2-c2595310cf2c.json
        """
        kernel = self.active_node.find_node_value_from_path(args.path)
        if not isinstance(kernel, KernelMetadata):
            raise ValueError(f"Could not find kernel located at {args.path}")
        self.send_stream(kernel.connection_file)

    def ls_cmd_handler(self, args):
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
        self.send_stream(target_node.childrens_to_str())

    def pwd_cmd_handler(self, args):
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
        self.send_stream(self.active_node.path)

    def complete_help_cmd(self, word, rank):
        return self.complete_cmd_name(word)

    def complete_kernels_cmd(self, word: str, rank):
        self.log.debug(f"Completing kernels : {word}, {rank}")
        runtime_dir = jupyter_runtime_dir()
        connection_files = glob.glob(os.path.join(runtime_dir, "kernel-*.json"))
        current_connection_file = get_connection_file()
        if current_connection_file in connection_files:
            connection_files.remove(current_connection_file)

        if word == "":
            return connection_files + self.get_available_kernels

        if word.startswith("/"):
            return self.complete_filesystem_path(word)
        else:
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
        self.log.debug(str(path_obj))
        if len(path) > 0 and path[-1] == "/":
            parent = path_obj
            name = ""
        else:
            parent = path_obj.parent
            name = path_obj.name

        self.log.debug(f"last element :{name}")
        self.log.debug(f"parent : {parent}")

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

        self.log.debug(all_matches)
        return all_matches

    def complete_cmd_name(self, cmd_name):
        self.log.debug(f"Completing cmd name {cmd_name}")
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
        self.log.debug(f"Completing path {path}")
        path_components = path.split("/")
        last_elem = path_components[-1]
        path_components.pop(-1)

        current_node = self.active_node
        if current_node is None:
            return []
        self.log.debug(f"Path components {path_components}, from node {current_node}")
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
        self.log.debug(f"Completing path : {word}")
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
        self.log.debug(f"Matches from local path : {matches}")
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

        new_kernel_parser = KomandParser("new")
        new_kernel_parser.add_argument("kernel", completer=self.complete_kernels_cmd)
        new_kernel_parser.add_argument("--label", "-l")
        new_kernel_cmd = SilikCommand(self.new_kernel_cmd_handler, new_kernel_parser)

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

        connection_file_parser = KomandParser("connect_info")
        connection_file_parser.add_argument(
            "path", completer=self.complete_local_path_arg
        )
        connection_file_cmd = SilikCommand(
            self.connection_file_cmd_handler, connection_file_parser
        )

        self.all_cmds: dict[str, SilikCommand] = {
            "kernels": kernels_cmd,
            "new": new_kernel_cmd,
            "restart": restart_cmd,
            "info": info_cmd,
            "connect_info": connection_file_cmd,
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

    def send_stream(self, text):
        """
        Prints raw text in the stdout of the kernel
        """
        self.send_response(
            self.iopub_socket,
            "stream",
            content={"name": "stdout", "text": text},
        )

    def send_out(self, text):
        """
        Sends an execute_result message, with text/plain mime type.
        """
        self.send_response(
            self.iopub_socket,
            "execute_result",
            {
                "execution_count": self.execution_count,
                "data": {"text/plain": text},
                "metadata": {},
            },
        )


class SubKernelExecutionDone(Exception):
    pass
