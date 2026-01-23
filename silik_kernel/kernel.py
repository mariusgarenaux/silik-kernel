# Basic python dependencies
import os
import traceback
from dataclasses import dataclass, field
from uuid import uuid4
import re
import random
from pathlib import Path
import logging
from typing import Literal, List, Optional, Callable
import shlex

# External dependencies
from ipykernel.kernelbase import Kernel
from jupyter_client.multikernelmanager import AsyncMultiKernelManager
from jupyter_client.kernelspec import KernelSpecManager


ALL_KERNELS_LABELS = [
    "lama",
    "loup",
    "kaki",
    "baba",
    "yack",
    "blob",
    "flan",
    "kiwi",
    "taco",
    "rose",
    "thym",
    "miel",
    "lion",
    "pneu",
    "lune",
    "ciel",
    "coco",
]
random.shuffle(ALL_KERNELS_LABELS)


def setup_kernel_logger(name, kernel_id, log_dir="~/.silik_logs"):
    """
    Creates a logger for the kernel. Set up SILIK_KERNEL_LOG environment
    variable to True before running the kernel, and create the following
    dir : ~/.silik_logs
    """
    log_dir = Path(log_dir).expanduser()
    if not os.path.isdir(log_dir):
        raise Exception(f"Please create a dir for kernel logs at {log_dir}")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fmt = logging.Formatter(
            f"%(asctime)s | {kernel_id[:5]} | %(levelname)s | %(name)s | %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


@dataclass
class KernelMetadata:
    """
    Custom dataclass used to describe kernels
    """

    label: str
    type: str
    id: str
    is_branched_to: "KernelMetadata | None" = None


@dataclass
class KernelTreeNode:
    """
    Stores the tree of kernels
    """

    value: KernelMetadata
    children: List["KernelTreeNode"] = field(default_factory=list)
    parent: Optional["KernelTreeNode"] = field(default=None)  # Add parent attribute

    def __post_init__(self):
        # Set the parent reference for children after initialization
        for child in self.children:
            child.parent = self

    def tree_to_str(self, pinned_node: KernelMetadata):
        def str_from_node(
            node: KernelTreeNode,
            prefix: str = "",
            is_last: bool = True,
            label_decorator="",
        ) -> str:
            # Initialize the representation of the tree as a list
            result = []

            # Append current node's label to the result
            displayed_label = (
                f"{label_decorator} {node.value.label} [{node.value.type}]"
                if node.value != pinned_node
                else f"{label_decorator} {node.value.label} [{node.value.type}] <<"
            )
            result.append(f"{prefix}{'╰─' if is_last else '├─'}{displayed_label}\n")

            # Determine the new prefix for child nodes
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterate over children and build the representation recursively
            for index, child in enumerate(node.children):
                if child.value == node.value.is_branched_to:
                    result.append(
                        str_from_node(
                            child,
                            new_prefix,
                            index == len(node.children) - 1,
                            ">",
                        ),
                    )
                else:
                    result.append(
                        str_from_node(
                            child, new_prefix, index == len(node.children) - 1
                        )
                    )

            return "".join(result)  # Join the list into a single string

        output = []
        for index, child in enumerate(self.children):
            output.append(
                str_from_node(
                    child,
                    prefix="",
                    is_last=index == len(self.children) - 1,
                )
            )

        return "".join(output)


class SilikCommandArgs:
    def __init__(self):
        pass


class SilikCommandParser:
    def __init__(
        self, positionals: list[str] | None = None, flags: list[str] | None = None
    ):
        self.positionals = positionals if positionals is not None else []
        self.flags = flags if flags is not None else []

    def parse(self, components):
        # Create an argument object
        arg_obj = SilikCommandArgs()
        for each_positional in self.positionals:
            arg_obj.__setattr__(each_positional, False)
        for each_flag in self.flags:
            arg_obj.__setattr__(each_flag, False)

        positional_idx = 0
        # Handle parameters and flags
        for component in components:
            if component.startswith("--"):
                # Handle flags
                if "=" in component:
                    key, value = component[2:].split("=", 1)
                    if key in self.flags:
                        arg_obj.__setattr__(key, value)
                    else:
                        raise ValueError(f"Unknown flag '{key}'")
                else:
                    key = component[2:]
                    if key in self.flags:
                        arg_obj.__setattr__(key, True)
                    else:
                        raise ValueError(f"Unknown flag '{key}'")
            else:
                arg_obj.__setattr__(
                    self.positionals[positional_idx], component
                )  # Store the value or process as needed
                positional_idx += 1

        return arg_obj


class SilikBaseKernel(Kernel):
    """
    Silik Kernel - Multikernel Manager

    Silik kernel is a gateway that distribute code cells towards
    different sub-kernels, e.g. :

    - octave
    - pydantic ai agent based kernel (https://github.com/mariusgarenaux/pydantic-ai-kernel)
    - python
    - an other silik-kernel !
    - ...

    See https://github.com/Tariqve/jupyter-kernels for available
    kernels.
    Silik kernel is a wrapper of MultiKernelManager in a jupyter kernel.

    The silik-kernel makes basic operations to properly start and
    stop sub-kernels, as well as providing helper functions to distribute
    code to sub-kernels.

    You should subclass this kernel in order to define custom strategies
    for :
        - sending messages (STDIN) to sub-kernels
        - merging outputs (STDOUT) and errors of sub-kernels outputs
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
        }

    async def get_kernel_history(self, kernel_id):
        """
        Returns the history of the kernel with kernel_id
        """
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
        specs = self.ksm.find_kernel_specs()
        return list(specs.keys())

    def get_kernel_from_label(self, kernel_label: str) -> KernelMetadata | None:
        for each_kernel in self.all_kernels:
            if each_kernel.label == kernel_label:
                return each_kernel

    def find_node_by_metadata(
        self, kernel_metadata: KernelMetadata
    ) -> KernelTreeNode | None:
        def recursively_find_node_in_tree(
            node: KernelTreeNode, kernel_metadata: KernelMetadata
        ):
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
        path_components = path.split("/")

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

    async def start_kernel(self, kernel_name: str, kernel_label: str | None = None):
        """
        Starts a kernel from its name (python3, ...)
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

    async def _do_execute(
        self,
        code: str,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        """
        Executes code on this kernel, without giving it to sub kernels.
        It is used to run commands, such as :
            - display active sub-kernels,
            - select kernel to run future code on,
            - start a kernel.
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

    async def do_execute(  # pyright: ignore
        self,
        code: str,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        try:
            # first checks for mode switch trigger (execution // transfer)
            first_word_trigger = code.split(" ")[0]
            if first_word_trigger in ["/cmd", "/cnct"]:
                self.logger.debug("Detected switch mode trigger")
                if first_word_trigger == "/cmd":
                    self.mode = "cmd"  # pyright: ignore
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

            # then either run code, or give it to sub-kernels according to a strategy
            if self.mode == "cmd":
                self.logger.debug(f"Command mode on {self.kernel_metadata.label}")
                result = await self._do_execute(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                return result
            elif self.mode == "cnct":
                self.logger.debug(f"Executing code on {self.active_kernel.label}")
                result = await self._do_connect(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                return result
            else:
                self.send_response(
                    self.iopub_socket,
                    "error",
                    {
                        "ename": "UnknownCommand",
                        "evalue": "",
                        "traceback": [""],
                    },
                )
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

    async def send_code_to_sub_kernel(
        self,
        sub_kernel: KernelMetadata,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        km = self.mkm.get_kernel(sub_kernel.id)
        kc = km.client()

        # synchronous call
        kc.start_channels()

        # msg_id = kc.execute(code)
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
            # metadata={"message_history": self.message_history},
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
        # synchronous call
        kc.stop_channels()
        if "error" in output:
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }, output["error"]

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

    async def _do_connect(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        """
        Transfer code to the sub-kernels. And sends the result through
        IOPubSocket.
        By default, code is sent to the selected kernel, but this behaviour
        could be modified.
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

        if result["status"] == "error":
            self.send_response(
                self.iopub_socket,
                "error",
                output,
            )
            return result
        if result["status"] == "ok":
            if not silent and output:
                for each_output_type in output:
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

    def parse_command(self, cell_input: str):
        """
        Parses the text to find a command. A command
        must start with !
        """
        parts = cell_input.split(" ", 1)  # Split the string at the first space
        first_word = parts[0]  # The first word
        rest_of_string = (
            parts[1] if len(parts) > 1 else ""
        )  # The rest of the string or empty if none
        return first_word, rest_of_string

    def do_shutdown(self, restart):
        self.mkm.shutdown_all()
        return super().do_shutdown(restart)

    # ------------------------------------------------------ #
    # ----------- COMMANDS HANDLERS AND PARSERS ------------ #
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
        if args.path is None:
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
        content = await self._do_connect(args.cmd, silent=False)
        if content["status"] == "error":
            return (
                True,
                f"[{self.kernel_metadata.label}] - Error during code execution",
            )
        return False, content


@dataclass
class SilikCommand:
    """
    ! GPT-5 generated !
    Lightweight command abstraction built on top of argparse.

    A Command binds an ArgumentParser to a handler function, allowing
    command-like execution without relying on argparse subparsers.
    """

    handler: Callable
    parser: SilikCommandParser = SilikCommandParser()
