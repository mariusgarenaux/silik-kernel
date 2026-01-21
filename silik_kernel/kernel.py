# Base python dependencies
import os
from dataclasses import dataclass, field
from uuid import uuid4
import random
from pathlib import Path
import logging
from typing import Literal, List, Optional

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
    "mite",
    "miel",
    "lion",
    "clou",
    "oeuf",
    "pneu",
    "lune",
    "ciel",
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
            node: KernelTreeNode, prefix: str = "", is_last: bool = True
        ) -> str:
            # Initialize the representation of the tree as a list
            result = []

            # Append current node's label to the result
            displayed_label = (
                f"{node.value.label} [{node.value.type}]"
                if node.value != pinned_node
                else f">> {node.value.label} << [{node.value.type}]"
            )
            result.append(f"{prefix}{'└─ ' if is_last else '├─ '}{displayed_label}\n")

            # Determine the new prefix for child nodes
            new_prefix = prefix + ("    " if is_last else "│   ")

            # Iterate over children and build the representation recursively
            for index, child in enumerate(node.children):
                result.append(
                    str_from_node(child, new_prefix, index == len(node.children) - 1)
                )

            return "".join(result)  # Join the list into a single string

        return str_from_node(self)


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
        self.mode: Literal["cmd", "run"] = "cmd"
        self.active_kernel: KernelMetadata = self.kernel_metadata
        self.root_node: KernelTreeNode = KernelTreeNode(
            self.kernel_metadata
        )  # stores the tree of all kernels

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

    async def start_kernel(self, kernel_name):
        self.logger.debug(f"Starting new kernel of type : {kernel_name}")
        kernel_id = str(uuid4())
        kernel_label = self.all_kernels_labels[self.kernel_label_rank]
        self.kernel_label_rank += 1
        new_kernel = KernelMetadata(label=kernel_label, type=kernel_name, id=kernel_id)
        await self.mkm.start_kernel(kernel_name=kernel_name, kernel_id=kernel_id)
        self.logger.debug(f"Successfully started kernel {new_kernel}")
        self.logger.debug(f"No kernel with label {kernel_name} is available.")
        return new_kernel

    async def _do_execute(
        self,
        code,
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
        out = self.parse_command(code)
        cmd, arg = out
        error = False
        match cmd:
            case "cd":
                if arg is None:  # cd
                    found_kernel = self.kernel_metadata
                else:
                    found_kernel = self.find_kernel_metadata_from_path(arg)
                if found_kernel is None:
                    error = True
                    content = f"Could not find kernel located at {arg}"
                else:
                    self.active_kernel = found_kernel
                    content = self.root_node.tree_to_str(self.active_kernel)
            case "mkdir":
                active_node = self.find_node_by_metadata(self.active_kernel)
                if active_node is None:
                    error = True
                    content = f"Could not find node in the kernel tree with value {self.active_kernel}"
                elif arg == "":
                    error = True
                    content = f"Please specify a kernel-type among {self.get_available_kernels}"
                else:
                    new_kernel = await self.start_kernel(arg)
                    active_node.children.append(
                        KernelTreeNode(new_kernel, parent=active_node)
                    )
                    content = self.root_node.tree_to_str(self.active_kernel)
            case "tree" | "ls":
                content = self.root_node.tree_to_str(self.active_kernel)
            # case "restart":
            #     found_kernel = self.get_kernel_with_label(arg)
            #     if found_kernel is None:
            #         content = f"Could not find kernel with label {arg}"
            #     else:
            #         await self.mkm.restart_kernel(found_kernel.id)
            #         content = f"Restarted kernel {found_kernel}"
            # case "ls":
            #     content = (
            #         f"{self.kernel_metadata.label}\n"
            #         if self.active_kernel != self.kernel_metadata
            #         else f">> {self.kernel_metadata.label} <<\n"
            #     )
            #     for k in range(len(self.all_sub_kernels)):
            #         knl = self.all_sub_kernels[k]
            #         label = (
            #             f">> {knl.label} <<" if knl == self.active_kernel else knl.label
            #         )
            #         dec = "╰──  " if k == len(self.all_sub_kernels) - 1 else "├──  "
            #         content += f"{dec}{label} [{knl.type}]\n"
            # case "pwd":
            #     if self.active_kernel is None:
            #         content = (
            #             "No kernel is running. Start one with `!start <kernel_name>`."
            #         )
            #     else:
            #         content = asdict(self.active_kernel)
            # case "help":
            #     content = "• !ls : prints living kernels\n• !start <kernel_type> : starts a kernel\n• !restart <kernel_label> : restart a kernel with its label\n• !select <kernel_label> : moves the selected kernel to the one with this label - nexts cells will be executed on this kernel\n• !kernels : list available kernels types"
            # case "kernels":
            #     content = self.get_available_kernels
            # case "history":
            #     content = await self.get_kernel_history(self.active_kernel.id)
            case _:
                error = True
                content = f"Unknown command {cmd}."

        if error:
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": "UnknownCommand",
                    "evalue": "Unknown command",
                    "traceback": [content],
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        self.send_response(
            self.iopub_socket,
            "execute_result",
            {
                "execution_count": self.execution_count,
                "data": {"text/plain": content},
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
            if first_word_trigger in ["/cmd", "/run"]:
                self.logger.debug("Detected switch mode trigger")
                if first_word_trigger == "/cmd":
                    self.mode = "cmd"  # pyright: ignore
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": self.execution_count,
                            "data": {
                                "text/plain": f"[{self.kernel_metadata.label}] is in mode {self.mode}. You can create and select kernels. Run help."
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
                    self.mode = "run"
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": self.execution_count,
                            "data": {
                                "text/plain": f"[{self.kernel_metadata.label}] acts as gateway towards {self.active_kernel}. All cells are executed on this kernel. Run /cmd to exit this mode and select a new kernel."
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
                self.logger.debug(f"Executing code on {self.kernel_metadata.label}")
                result = await self._do_execute(
                    code, silent, store_history, user_expressions, allow_stdin
                )
                return result
            elif self.mode == "run":
                self.logger.debug(f"Executing code on {self.active_kernel.label}")
                result = await self._do_run(
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
            self.send_response(
                self.iopub_socket,
                "error",
                {
                    "ename": str(e),
                    "evalue": str(e),
                    "traceback": [str(e.__traceback__)],
                },
            )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

    async def _do_run(
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
        km = self.mkm.get_kernel(self.active_kernel.id)
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
        output = []

        while True:
            msg = await kc._async_get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]

            if msg_type == "execute_result":
                output = msg["content"]["data"]["text/plain"]
                break

            elif msg_type == "error":
                self.logger.debug(f"Error : {msg}")
                kc.stop_channels()

                self.send_response(
                    self.iopub_socket,
                    "error",
                    {
                        "ename": msg["content"]["ename"],
                        "evalue": msg["content"]["evalue"],
                        "traceback": msg["content"]["traceback"],
                    },
                )
                return {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "payload": [],
                    "user_expressions": {},
                }

            elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                break

        # synchronous call
        kc.stop_channels()

        if not silent and output:
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

    def parse_command(self, cell_input: str):
        """
        Parses the text to find a command. A command
        must start with !.
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
