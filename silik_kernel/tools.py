# Internal dependencies

from .types import IOPubMsg, ExecutionResult

# Basic python dependencies
import os
from dataclasses import dataclass, field
import random
from pathlib import Path
import logging
from typing import Literal, List, Optional, Callable, Annotated, Tuple, Self
from statikomand import KomandParser
from argparse import Namespace

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
    logging_level_env = os.getenv("SILIK_KERNEL_LOG_LEVEL")
    logging_level_str = logging_level_env if logging_level_env is not None else "DEBUG"
    logging_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(logging_level_str, logging.DEBUG)

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fmt = logging.Formatter(
            f"%(asctime)s | {kernel_id[:5]} | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


pretty_display = os.getenv("SILIK_KERNEL_PRETTY_DISPLAY")
PRETTY_DISPLAY = True if pretty_display in ["True", "true", "1", 1] else False


@dataclass
class NodeValue:
    """
    Value attached to a node of the tree (either a
    directory, or a kernel metadata).
    """

    id: str
    label: str

    def __str__(self) -> str:
        return self.label


@dataclass
class KernelMetadata(NodeValue):
    """
    Custom dataclass used to describe kernels
    """

    id: str
    type: str
    kernel_name: str
    label: str
    kernel_info: Optional[dict] = field(default=None)
    connection_file: Optional[str] = field(default=None)


@dataclass
class KernelFolder(NodeValue):
    """
    Dataclass that describes a folder of kernels
    """

    label: str
    id: str

    def __str__(self) -> str:
        if PRETTY_DISPLAY:
            return f"\033[1;96m{self.label}\033[0m"
        else:
            return self.label


@dataclass
class TreeNode:
    """
    Stores the tree of directories and kernels.
    """

    value: NodeValue
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = field(default=None)  # Add parent attribute

    @property
    def node_type(self):
        if isinstance(self.value, KernelFolder):
            return "folder"
        if isinstance(self.value, KernelMetadata):
            return "leaf"

    def __post_init__(self):
        # Set the parent reference for children after initialization
        for child in self.children:
            child.parent = self

    def childrens_to_str(self):
        output = ""
        for child in self.children:
            output += f"{child.value}\n"

        return output

    def tree_to_str(self, pinned_node: NodeValue | None = None):
        def str_from_node(
            node: TreeNode,
            prefix: str = "",
            is_last: bool = True,
            label_decorator="",
        ) -> str:
            # Initialize the representation of the tree as a list
            result = []

            displayed_label = f"{label_decorator} {node.value}"

            result.append(f"{prefix}{'╰─' if is_last else '├─'}{displayed_label}\n")

            # Determine the new prefix for child nodes
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterate over children and build the representation recursively
            for index, child in enumerate(node.children):
                result.append(
                    str_from_node(child, new_prefix, index == len(node.children) - 1)
                )

            return "".join(result)  # Join the list into a single string

        output = [f"{self.value}\n"]
        for index, child in enumerate(self.children):
            output.append(
                str_from_node(
                    child,
                    prefix="",
                    is_last=index == len(self.children) - 1,
                )
            )

        return "".join(output)

    def find_node_by_value(self, node_value: NodeValue) -> "TreeNode | None":
        """
        Find the node in the tree which has of the node value.

        Parameters:
        ---

            - kernel_tree_value (KernelTreeValue): the metadata describing either
                the kernel or the folder containing the kernel.

        Returns:
        ---

            The node object (KernelTreeNode), or None if no match was found
        """

        def recursively_find_node_in_tree(node: TreeNode, kernel_tree_value: NodeValue):
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
            if node.value == kernel_tree_value:
                return node

            # Recursively search through children
            for child in node.children:
                found_node = recursively_find_node_in_tree(child, kernel_tree_value)
                if found_node:
                    return found_node  # Return the found node if it exists

            return (
                None  # Return None if the target metadata is not found in the subtree
            )

        return recursively_find_node_in_tree(self, node_value)

    def find_node_value_from_path(self, path: str) -> NodeValue | None:
        """
        Finds a node value from its path starting from the current dir

        Parameters:
        ---

            - path: the relative path in posix fashion (e.g. dir_1/kernel_2
                or ../dir_1/kernel_2) from the active kernel

        Returns:
        ---
            The KernelMetadata located at <path> from active kernel; or None if
            path is incorrect.
        """
        tree_node = self.find_node_from_path(path)
        if tree_node is None:
            return
        return tree_node.value

    def find_node_from_path(self, path: str) -> "TreeNode | None":
        path_components = path.split("/")
        if path_components[-1] == "":
            path_components.pop(-1)

        current_node = self

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

        return current_node  # Return the KernelMetadata of the found node

    def add_children(self, node_value: NodeValue) -> "TreeNode":
        if self.node_type != "folder":
            raise ValueError(
                "Can not create a children to a leaf of the tree. Ensure the node is a directory."
            )
        children = TreeNode(node_value, parent=self)
        self.children.append(children)
        return children

    @property
    def path(self):
        out = ""
        parent_node = self
        while True:
            out = parent_node.value.label + "/" + out
            if parent_node.parent is None:
                break
            parent_node = parent_node.parent
        return out


class SilikCommandArgs:
    def __init__(self):
        pass


@dataclass
class SilikCommand:
    handler: Annotated[
        Callable[[Namespace], Tuple[ExecutionResult, IOPubMsg]],
        "Method that is called to run the command. Take as input ParsedKomandArgs. Must output a Tuple (ExecutionResult, IOPubMsg)",
    ]
    parser: KomandParser
