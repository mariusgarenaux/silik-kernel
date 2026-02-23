# Internal dependencies

from .types import IOPubMsg, ExecutionResult

# Basic python dependencies
import os
from dataclasses import dataclass, field
import random
from pathlib import Path
import logging
from typing import Literal, List, Optional, Callable, Annotated, Tuple
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

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fmt = logging.Formatter(
            f"%(asctime)s | {kernel_id[:5]} | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
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
    connection_file: Optional[str] = None


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
                result.append(
                    str_from_node(child, new_prefix, index == len(node.children) - 1)
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


@dataclass
class SilikCommand:
    handler: Annotated[
        Callable[[Namespace], Tuple[ExecutionResult, IOPubMsg]],
        "Method that is called to run the command. Take as input ParsedKomandArgs. Must output a Tuple (ExecutionResult, IOPubMsg)",
    ]
    parser: KomandParser
