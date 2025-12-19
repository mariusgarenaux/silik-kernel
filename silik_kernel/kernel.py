# Base python dependencies
import re
import os
from dataclasses import dataclass, asdict
from uuid import uuid4
import random
from pathlib import Path
import logging

# External dependencies
from ipykernel.kernelbase import Kernel
from jupyter_client.multikernelmanager import AsyncMultiKernelManager


ALL_KERNELS_LABELS = [
    "lion",
    "tiger",
    "bear",
    "wolf",
    "deer",
    "fox",
    "cat",
    "dog",
    "bird",
    "fish",
    "horse",
    "cow",
    "pig",
    "sheep",
    "goat",
    "duck",
    "chicken",
    "turkey",
    "frog",
    "toad",
    "snake",
    "lizard",
    "turtle",
    "rabbit",
    "mouse",
    "rat",
    "bat",
    "frog",
    "fern",
    "ivy",
    "lily",
    "rose",
    "sage",
    "thyme",
    "mint",
    "basil",
    "parsley",
    "daisy",
    "sunflower",
    "violet",
    "poppy",
    "orchid",
    "jasmine",
    "heather",
    "lavender",
    "geranium",
    "begonia",
    "camellia",
    "azalea",
    "gardenia",
    "hibiscus",
    "magnolia",
    "petunia",
    "primrose",
    "snapdragon",
    "sweetpea",
    "tulip",
    "veronica",
    "oak",
    "pine",
    "maple",
    "elm",
    "ash",
    "beech",
    "birch",
    "cherry",
    "cypress",
    "ebony",
    "fir",
    "hazel",
    "juniper",
    "larch",
    "mahogany",
    "mulberry",
    "olive",
    "palm",
    "pecan",
    "poplar",
    "quaking",
    "redwood",
    "spruce",
    "sycamore",
    "teak",
    "walnut",
    "willow",
    "yew",
    "bee",
    "ant",
    "fly",
    "moth",
    "wasp",
    "bug",
    "flea",
    "tick",
    "mite",
    "louse",
    "roach",
    "cricket",
    "grasshopper",
    "cicada",
    "katydid",
    "earwig",
    "silverfish",
    "spider",
    "scorpion",
    "centipede",
    "millipede",
    "dragonfly",
    "damselfly",
    "butterfly",
    "ladybug",
    "firefly",
    "stick",
    "aphid",
    "termite",
    "beet",
    "carrot",
    "potato",
    "onion",
    "garlic",
    "leek",
    "radish",
    "turnip",
    "parsnip",
    "celery",
    "kale",
    "spinach",
    "peas",
    "cucumber",
    "eggplant",
    "okra",
    "squash",
    "pumpkin",
    "corn",
    "broccoli",
    "cauliflower",
    "mushroom",
    "asparagus",
    "fennel",
    "rhubarb",
    "zucchini",
    "yam",
    "ginger",
    "horseradish",
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "guava",
    "jackfruit",
    "kiwi",
    "lemon",
    "mango",
    "melon",
    "nectarine",
    "orange",
    "papaya",
    "pear",
    "pineapple",
    "pomegranate",
    "quince",
    "raspberry",
    "strawberry",
    "tangerine",
    "ugli",
    "victoria",
    "watermelon",
    "xigua",
    "yuzu",
    "ziziphus",
]


def setup_kernel_logger(name, log_dir="~/.silik_logs"):
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
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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


class SilikBaseKernel(Kernel):
    """
    Base Kernel for Silik, is used as a gateway to distribute
    code cells towards different kernels, e.g. :

    - octave
    - pydantic ai agent based kernel (https://github.com/mariusgarenaux/pydantic-ai-kernel)
    - python
    - ...

    See https://github.com/Tariqve/jupyter-kernels for available
    kernels.
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
    banner = "Silik Kernel - Multikernel Manager - Send !help for commands"
    active_kernel: KernelMetadata | None = None
    all_kernels: list[KernelMetadata] = []
    all_kernels_labels: list[str] = ALL_KERNELS_LABELS
    mkm: AsyncMultiKernelManager = AsyncMultiKernelManager()
    message_history: list[dict] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        should_custom_log = os.environ.get("SILIK_KERNEL_LOG", "False")
        should_custom_log = (
            True if should_custom_log in ["True", "true", "1"] else False
        )

        if should_custom_log:
            logger = setup_kernel_logger(__name__)
            logger.debug("Started kernel and initalized logger")
            self.logger = logger
        else:
            self.logger = self.log

    def get_kernel_with_id(self, kernel_id: str) -> KernelMetadata | None:
        """
        Returns a kernel metadata from its id.

        Parameters :
        ---
            - kernel_id : the uuidv4 (str) of the wanted kernel

        Returns :
        ---
        The matching kernel metadata if found, else None.
        """
        for each_kernel in self.all_kernels:
            if each_kernel.id == kernel_id:
                return each_kernel

    def get_kernel_with_label(self, kernel_label: str) -> KernelMetadata | None:
        """
        Returns a kernel metadata from its label.

        Parameters :
        ---
            - kernel_label : the label (str) of the wanted kernel

        Returns :
        ---
        The matching kernel metadata if found, else None.
        """
        for each_kernel in self.all_kernels:
            if each_kernel.label == kernel_label:
                return each_kernel

    async def _do_execute(
        self, code, silent, store_history, user_expressions, allow_stdin
    ):
        """
        Custom do_execute function that :

            - retrieve the uuid of the living kernel
            - pass the code to it, and displays its result
        """
        self.logger.debug(f"Active kernel : {self.active_kernel}")
        if self.active_kernel is None:
            self.send_response(
                self.iopub_socket,
                "display_data",
                {
                    "data": {
                        "text/plain": "Please start a kernel. Run '!start <kernel_type>'"
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
            metadata={"message_history": self.message_history},
        )
        kc.shell_channel.send(msg)
        msg_id = msg["header"]["msg_id"]
        output = []

        while True:
            msg = await kc._async_get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]

            if msg_type == "stream":
                output.append(msg["content"]["text"])

            elif msg_type == "execute_result":
                output.append(msg["content"]["data"]["text/plain"])

            elif msg_type == "error":
                output.append("\n".join(msg["content"]["traceback"]))
                self.logger.debug(f"Error : {msg}")
                return {
                    "status": "error",
                    "ename": msg["content"]["ename"],
                    "evalue": msg["content"]["evalue"],
                    "traceback": msg["content"]["traceback"],
                }
            elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                break

        # synchronous call
        kc.stop_channels()

        if not silent and output:
            self.send_response(
                self.iopub_socket,
                "display_data",
                {
                    "data": {"text/plain": output[0]},
                    "metadata": {},
                },
            )

            self.message_history.append(
                {"role": "user", "content": code, "uid": str(uuid4())}
            )
            self.message_history.append(
                {"role": "assistant", "content": output[0], "uid": str(uuid4())}
            )

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def select_kernel(self, kernel_label: str) -> KernelMetadata | None:
        """
        Moves the selected kernel to kernel_label.
        """
        found_kernel = self.get_kernel_with_label(kernel_label)
        if found_kernel is not None:
            self.active_kernel = found_kernel
            return found_kernel

    async def start_kernel(self, kernel_name: str) -> KernelMetadata | None:
        """
        Finds a living kernel and returns its ID. If it does not exists,
        starts it.

        Returns :
        ---
            KernelMetadata or None
        """
        self.logger.debug(f"Starting new kernel of type : {kernel_name}")
        kernel_id = await self.mkm.start_kernel(kernel_name=kernel_name)
        kernel_label = random.choice(self.all_kernels_labels)
        new_kernel = KernelMetadata(label=kernel_label, type=kernel_name, id=kernel_id)
        self.active_kernel = new_kernel
        self.logger.debug(f"Successfully started kernel {new_kernel}")
        self.all_kernels.append(new_kernel)
        self.logger.debug(f"No kernel with label {kernel_name} is available.")
        return new_kernel

    async def do_execute(  # pyright: ignore
        self,
        code: str,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        try:
            if not silent:
                out = self.parse_command(code)
                self.logger.debug(f"Parsed {out} from {code}")
                if out is None:
                    result = await self._do_execute(
                        code, silent, store_history, user_expressions, allow_stdin
                    )
                    return result
                cmd, arg = out
                match cmd:
                    case "start":
                        found_kernel = await self.start_kernel(arg)
                        if found_kernel is None:
                            content = f"Could not create kernel with name {arg}"
                        else:
                            content = f"Connected to kernel {found_kernel}. Now, cells are executed on this kernel."
                    case "select":
                        found_kernel = self.select_kernel(arg)
                        if found_kernel is None:
                            content = f"Could not find kernel with name {arg}"
                        else:
                            content = f"Connected to kernel {found_kernel}. Now, cells are executed on this kernel."
                    case "restart":
                        found_kernel = self.get_kernel_with_label(arg)
                        if found_kernel is None:
                            content = f"Could not find kernel with label {arg}"
                        else:
                            await self.mkm.restart_kernel(found_kernel.id)
                            content = f"Restarted kernel {found_kernel}"
                    case "ls":
                        content = "silik\n"
                        for k in range(len(self.all_kernels)):
                            knl = self.all_kernels[k]
                            label = (
                                f">> {knl.label} <<"
                                if knl == self.active_kernel
                                else knl.label
                            )
                            dec = "╰──  " if k == len(self.all_kernels) - 1 else "├──  "
                            content += f"{dec}{label} [{knl.type}]\n"
                    case "pwd":
                        if self.active_kernel is None:
                            content = "No kernel is running. Start one with `!start <kernel_name>`."
                        else:
                            content = asdict(self.active_kernel)
                    case "help":
                        content = "• !ls : prints living kernels\n• !start <kernel_type> : starts a kernel\n• !restart <kernel_label> : restart a kernel with its label\n• !select <kernel_label> : moves the selected kernel to the one with this label - nexts cells will be executed on this kernel"
                    case _:
                        content = f"Unknown command {cmd}."

                self.send_response(
                    self.iopub_socket,
                    "execute_result",
                    {
                        "execution_count": self.execution_count,
                        "data": {"text/plain": content},
                        "metadata": {},
                    },
                )
        except Exception as e:
            # TODO : send to error socket
            content = str(e)
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

    def parse_command(self, cell_input: str):
        """
        Parses the text to find a command. A command
        must start with !.
        """
        matched = re.match(r"^!(\w+)(?: ([\w.]+))?$", cell_input)
        try:
            cmd_name = matched.group(1)  # Extract the command name # pyright: ignore
            argument = matched.group(2)  # Extract the argument # pyright: ignore
            return cmd_name, argument
        except Exception as e:
            self.logger.debug(f"Could not parse info on {cell_input}, because {e}")
            return

    def do_shutdown(self, restart):
        self.mkm.shutdown_all()
        return super().do_shutdown(restart)
