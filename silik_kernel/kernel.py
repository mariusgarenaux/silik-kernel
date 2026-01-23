# Basic python dependencies
import os
import traceback
from uuid import uuid4, UUID
import re
from typing import Literal, Optional, Tuple
import shlex

# Internal dependencies
from .tools import (
    ALL_KERNELS_LABELS,
    setup_kernel_logger,
    ExecutionResult,
    KernelMetadata,
    KernelTreeNode,
    SilikCommand,
    SilikCommandParser,
)

# External dependencies
from ipykernel.kernelbase import Kernel
from jupyter_client.multikernelmanager import AsyncMultiKernelManager
from jupyter_client.kernelspec import KernelSpecManager


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
    code to sub-kernels.

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
    implementation_version = "1.0"
    language = "no-op"
    language_version = "0.1"
    language_info = {
        "name": "silik",
        "mimetype": "text/plain",
        "file_extension": ".txt",
    }
    banner = "Silik Kernel - Multikernel Manager - Run `help` for commands"
    all_kernels_labels: list[str] = ALL_KERNELS_LABELS
    mkm: AsyncMultiKernelManager = AsyncMultiKernelManager()
    message_history: dict[str, list] = {}  # history of messages

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

        ls_cmd = SilikCommand(self.ls_cmd_handler)
        run_cmd = SilikCommand(self.run_cmd_handler, SilikCommandParser(["cmd"]))
        self.all_cmds: dict[str, SilikCommand] = {
            "cd": SilikCommand(self.cd_cmd_handler, SilikCommandParser(["path"])),
            "mkdir": SilikCommand(
                self.mkdir_cmd_handler, SilikCommandParser(["kernel_type"], ["label"])
            ),
            "ls": ls_cmd,
            "tree": ls_cmd,
            "restart": SilikCommand(self.restart_cmd_handler),
            "kernels": SilikCommand(self.kernels_cmd_handler),
            "history": SilikCommand(self.history_cmd_handler),
            "branch": SilikCommand(
                self.branch_cmd_handler, SilikCommandParser(["kernel_label"])
            ),
            "detach": SilikCommand(self.detach_cmd_handler),
            "run": run_cmd,
            "r": run_cmd,
            "help": SilikCommand(self.help_cmd_handler),
            "exit": SilikCommand(
                self.exit_cmd_handler, SilikCommandParser(flags=["restart"])
            ),
        }

    # ------------------------------------------------------ #
    # ---------------- do_execute methods ------------------ #
    # ------------------------------------------------------ #

    async def do_execute(  # pyright: ignore
        self,
        code: str,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
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
            ExecutionResult, according to IPykernel documentation.
        """
        try:
            # first checks for mode switch trigger (execution // transfer)
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
                self.logger.debug(f"Command mode on {self.kernel_metadata.label}")
                result = await self.do_execute_on_silik(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                return result
            elif self.mode == "cnct":
                self.logger.debug(f"Executing code on {self.active_kernel.label}")
                result = await self.do_execute_on_sub_kernel(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                return result
            else:
                self.mode = "cmd"
                return {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }
        except Exception as e:
            traceback_list = traceback.format_exc().splitlines()
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": str(e),
                    "evalue": str(e),
                    "traceback": traceback_list,
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

    async def do_execute_on_silik(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> ExecutionResult:
        """
        Executes code on this kernel, without giving it to sub kernels.
        Sends content to IOPub channel. Accepted code contains bash like
        commands, that is parsed and executed according to self.all_cmds
        dictionnary.

        Example commands actions :
            - display tree,
            - display active sub-kernels,
            - select kernel to run future code on,
            - start a kernel,
            - running one-line code on sub kernels.

        Parameters:
        ---

            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g. for Python’s raw_input()).

        """
        splitted = code.split(" ", 1)
        self.logger.debug(splitted)
        if len(splitted) == 0:
            self.logger.debug(f"Splitted is empty {splitted}")
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "UnknownCommand",
                    "evalue": "Unknown command",
                    "traceback": ["Could not parse command"],
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }
        cmd_name = splitted[0]
        if re.fullmatch(r"r\d", cmd_name):
            num_it = int(cmd_name[1])
            cmd_name = "r"
            splitted[0] = "r"
            splitted[1] = "r " * (num_it - 1) + splitted[1]

        self.logger.debug(f"Command {cmd_name} | Splitted {splitted}")
        if cmd_name not in self.all_cmds:
            self.logger.debug(f"Cmd not found {cmd_name}")
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "UnknownCommand",
                    "evalue": "Unknown command",
                    "traceback": ["Could not parse command"],
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }
        cmd_obj = self.all_cmds[cmd_name]
        self.logger.debug(f"Cmd obj {cmd_obj}")
        if cmd_name not in ["run", "r"]:
            splitted = shlex.split(code)
            self.logger.debug(f"shlex splitted {splitted}")

        args = cmd_obj.parser.parse(splitted[1:])
        self.logger.debug(f"_do_execute {cmd_name} {args}, {cmd_obj.handler}")
        cmd_out = await cmd_obj.handler(args)
        self.logger.debug(f"here cmd_out {cmd_out}")
        if cmd_out is None:
            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }
        error, output = cmd_out

        if error:
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "UnknownCommand",
                    "evalue": "Unknown command",
                    "traceback": [output],
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }
        if cmd_name not in ["r", "run"]:
            self.send_response(
                self.iopub_socket,
                "execute_result",
                {
                    "execution_count": self.execution_count,
                    "data": {"text/plain": output},
                    "metadata": {},
                },
            )

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    async def do_execute_on_sub_kernel(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> ExecutionResult:
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

        """
        self.logger.debug(f"Code is sent to selected kernel : {self.active_kernel}")

        result, output = await self.send_code_to_sub_kernel(
            self.active_kernel,
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
        )
        self.logger.debug(f"Output of cell : {result, output}")

        if silent:
            return result

        for each_output_type in output:
            # sends content to each channel (error, execution result, ...)
            self.logger.debug(f"Output type {output[each_output_type]}")
            if each_output_type == "execute_result":
                output[each_output_type]["data"]["text/plain"] = (
                    f"{self.active_kernel.label} [{self.active_kernel.type}]\n"
                    + output[each_output_type]["data"]["text/plain"]
                )
            self.send_response(
                self.iopub_socket,
                each_output_type,
                output[each_output_type],
            )

        return result

    # ------------------------------------------------------ #
    # ------------- tools for executing code --------------- #
    # ------------------------------------------------------ #

    async def get_kernel_history(self, kernel_id: UUID | str) -> list:
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
                msg = await kc._async_get_shell_msg()
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
        return list(specs.keys())

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
            The KernelMetadata object describing the kernel.
        """
        self.logger.debug(f"Starting new kernel of type : {kernel_name}")
        kernel_id = str(uuid4())
        if kernel_label is None:
            kernel_label = self.all_kernels_labels[self.kernel_label_rank]
            self.kernel_label_rank += 1
        new_kernel = KernelMetadata(label=kernel_label, type=kernel_name, id=kernel_id)
        await self.mkm.start_kernel(kernel_name=kernel_name, kernel_id=kernel_id)
        self.logger.debug(f"Successfully started kernel {new_kernel}")
        self.logger.debug(f"No kernel with label {kernel_name} is available.")
        return new_kernel

    async def send_code_to_sub_kernel(
        self,
        sub_kernel: KernelMetadata,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
    ) -> Tuple[
        ExecutionResult, dict[Literal["error", "display_data", "execute_result"], dict]
    ]:
        """
        Send code to sub kernel for execution through shell channel.
        Code is sent recursively if the sub-kernel is branched to one
        of its children. Returns a tuple :
        (execution_result, dict[str, message_content]).

        The keys of the second element are message types (from the IOPub channel),
        among :
            - error,
            - display_data,
            - execute_result.

        Values are the content of messages of this type, and are directly those of
        sub-kernel (see https://jupyter-client.readthedocs.io/en/stable/messaging.html)
        For example, for message type execute_result, the value follows this scheme :
            content = {
                'execution_count': int,
                'data' : dict,
                'metadata' : dict,
            }

        The last parameters are those asked by the IPykernel wrapper method
        do_execute :
        https://jupyter-client.readthedocs.io/en/stable/wrapperkernels.html#MyKernel.do_execute

        Parameters
        ---

            - sub_kernel : A KernelMetadata object describing the sub kernel
            - code (str) : The code to be executed.
            - silent (bool) : Whether to display output.
            - store_history (bool) : Whether to record this code in history and increase the execution count. If silent is True, this is implicitly False.
            - user_expressions (dict) : Mapping of names to expressions to evaluate after the code has run. You can ignore this if you need to.
            - allow_stdin (bool) : Whether the frontend can provide input on request (e.g. for Python’s raw_input()).


        """
        km = self.mkm.get_kernel(sub_kernel.id)
        kc = km.client()
        kc.start_channels()
        content = {
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
        output = {}

        while True:
            msg = await kc._async_get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            self.logger.debug(f"msg from kernel {msg}")
            if msg_type == "execute_result":
                output["execute_result"] = msg["content"]
            elif msg_type == "stream":
                output["stream"] = msg["content"]
            elif msg_type == "display_data":
                output["display_data"] = msg["content"]

            elif msg_type == "error":
                output["error"] = msg["content"]
                break

            elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                break
            if "execute_result" in output:
                break

        kc.stop_channels()
        if "error" in output:
            # we stop recursion if one error is caught
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }, output

        if sub_kernel.is_branched_to is not None and "execute_result" in output:
            raw_output = output["execute_result"]["data"]["text/plain"]
            self.logger.debug(
                f"Sending output : `{raw_output}` of {sub_kernel.label} to {sub_kernel.is_branched_to.label}"
            )
            res = await self.send_code_to_sub_kernel(
                sub_kernel=sub_kernel.is_branched_to,
                code=raw_output,
                silent=silent,
            )
            self.logger.debug(res)
            return res
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }, output

    def do_shutdown(self, restart: bool):
        """
        Shutdown the kernel by simply shutting down all sub kernels
        started with
        """

        self.mkm.shutdown_all()
        # TODO: mkm shutdown is async, but here it is sync
        return super().do_shutdown(restart)

    # ------------------------------------------------------ #
    # ----------------- COMMANDS HANDLERS ------------------ #
    # ------------------------------------------------------ #

    async def help_cmd_handler(self, args):
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

        return False, content

    async def cd_cmd_handler(self, args):
        error = False
        if args.path is None or not args.path:
            found_kernel = self.kernel_metadata
        else:
            found_kernel = self.find_kernel_metadata_from_path(args.path)
        if found_kernel is None:
            error = True
            content = f"Could not find kernel located at {args.path}"
            return error, content
        self.active_kernel = found_kernel
        content = self.root_node.tree_to_str(self.active_kernel)
        return error, content

    async def mkdir_cmd_handler(self, args):
        self.logger.debug(f"mkdir with args {args}")
        error = False
        active_node = self.find_node_by_metadata(self.active_kernel)
        self.logger.debug(active_node)
        if active_node is None:
            error = True
            content = f"Could not find node in the kernel tree with value {self.active_kernel}"
            return error, content
        if not args.kernel_type:
            error = True
            content = f"Please specify a kernel-type among {self.get_available_kernels}"
            self.logger.debug(f"{error, content}")
            return error, content

        new_kernel = await self.start_kernel(
            args.kernel_type, None if not args.label else args.label
        )
        active_node.children.append(KernelTreeNode(new_kernel, parent=active_node))
        self.all_kernels.append(new_kernel)
        content = self.root_node.tree_to_str(self.active_kernel)
        return error, content

    async def ls_cmd_handler(self, args):
        content = self.root_node.tree_to_str(self.active_kernel)
        self.logger.debug(f"ls here {content}")
        return False, content

    async def restart_cmd_handler(self, args):
        await self.mkm.restart_kernel(self.active_kernel.id)
        content = f"Restarted kernel {self.active_kernel.label}"
        return False, content

    async def kernels_cmd_handler(self, args):
        content = self.get_available_kernels
        return False, content

    async def history_cmd_handler(self, args):
        content = await self.get_kernel_history(self.active_kernel.id)
        return False, content

    async def branch_cmd_handler(self, args):
        # connects output of selected kernel to one kernel
        out_kernel = self.get_kernel_from_label(args.kernel_label)
        if out_kernel is None:
            error = True
            content = f"No kernel was found with label `{args.kernel_label}`"
            return error, content

        active_kernel_node = self.find_node_by_metadata(self.active_kernel)
        if active_kernel_node is None:
            error = True
            content = (
                f"No kernel was found on tree with label {self.active_kernel.label}."
            )
            return error, content

        out_kernel_node = self.find_node_by_metadata(out_kernel)
        if out_kernel_node is None:
            error = True
            content = f"No kernel was found on tree with label {out_kernel.label}."
            return error, content

        if out_kernel_node not in active_kernel_node.children:
            error = True
            content = f"Kernel {out_kernel.label} not in {self.active_kernel.label} childrens. Branching is only available from a parent to a children."
            return error, content

        self.active_kernel.is_branched_to = out_kernel
        content = self.root_node.tree_to_str(self.active_kernel)
        return False, content

    async def detach_cmd_handler(self, args):
        self.active_kernel.is_branched_to = None
        content = self.root_node.tree_to_str(self.active_kernel)
        return False, content

    async def run_cmd_handler(self, args):
        content = await self.do_execute_on_sub_kernel(args.cmd, silent=False)
        if content["status"] == "error":
            return (
                True,
                f"[{self.kernel_metadata.label}] - Error during code execution",
            )
        return False, content

    async def exit_cmd_handler(self, args):
        return True, "Not implemented"
