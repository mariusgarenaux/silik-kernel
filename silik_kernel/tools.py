# Basic python dependencies
import os
from dataclasses import dataclass, field
import random
from pathlib import Path
import logging
from typing import Literal, List, Optional, Callable, TypedDict
from statikomand import KomandParser

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
    is_branched_to: "KernelMetadata | None" = None
    branch_trust_level: Literal[0, 1, 2] = 0  # trust level of branched kernel
    # 0 = no trust, each cell is asked for validation before being sent to the sub kernel
    # 1 = trust but watch, each cell is displayed to the user but is still automatically sent to the sub kernel
    # 2 = full trust, each cell is automatically sent to the sub-kernel, and not sent to the user
    sandbox: dict = field(default_factory=dict)


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


@dataclass
class SilikCommand:
    handler: Callable
    parser: KomandParser
