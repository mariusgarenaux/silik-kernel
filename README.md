# Silik Kernel

> A Jupyter Multi Kernel Manager, wrapped in a Jupyter Kernel 🙂 Because "who is more qualified to relay kernel messages than a kernel itself ?"

![](example.mp4)

This is a jupyter kernel that allows to interface with multiple kernels, you can:

- start, stop and restart kernels,

- switch between kernels,

- list available kernels,

- connect to a living kernel,

- "store" kernels in fictive "directories"

> All of this with simple command lines

A Silik Kernel can be in two modes :

- **`command`** mode : manage kernels (start, stop, ...),

- **`connect`** mode : connects to one kernel, and acts as a gateway with this kernel. Support for TAB completion, and propagation of most sockets messages is implemented here.

> **Any jupyter kernel can be accessed through silik-kernel**

> But managing interaction between kernels seems to be a nightmare ?

**Not with Agents and LLM**. In order to allow users to easily manage multi-kernels, we present a way to access AI agents through jupyter kernels. To do so, we provide a [wrapper of a pydantic-ai agent in a kernel](https://github.com/mariusgarenaux/pydantic-ai-kernel). This allows to interact easily with these agents, through ipython for example, and let them manage the output of cells.

It also allows to share agents easily (with **pypi** for example); because they can be shipped in a python module. We split properly the agent and the interaction framework with the agent, by reusing the ones from jupyter kernels.

## Getting started

```bash
pip install silik-kernel
```

The kernel is then installed on the current python venv.

Any jupyter frontend should be able to access the kernel, for example :

• **Notebook** (you might need to restart the IDE) : select 'silik' on top right of the notebook

• **CLI** : Install jupyter-console (`pip install jupyter-console`); and run `jupyter console --kernel silik`

• **Silik Signal Messaging** : Access the kernel through Signal Message Application.

To use diverse kernels through silik, you can install some example kernels : [https://github.com/Tariqve/jupyter-kernels](https://github.com/Tariqve/jupyter-kernels). You can also create new agent-based kernel by subclassing [pydantic-ai base kernel](https://github.com/mariusgarenaux/pydantic-ai-kernel).

> You can list the available kernels by running `jupyter kernelspec list` in a terminal. Or with TAB completion of `start` command in a cell of a silik kernel.

## Usage

### Example

Install and start silik-kernel :

```bash
python -m venv .venv
source .venv/bin/activate
pip install silik-kernel
jupyter console --kernel silik
```

Within a cell :

```bash
start python3 --label k1
run "x = 19" k1.py
run "print(x)" k1.py
```

For persistent connection :

```bash
> k1.py
```

Now, all cells of silik kernel are directly transmitted to kernel k1.py. Try:

```bash
print(x)
```

### Usage Guide

To go from `command` mode to `connect` mode, you have to use the command :

```bash
> path_to_kernel
```

> Since this changes the language of the following cells, it must not be mixed with the language of the kernel you connects to. The `> path_to_kernel` needs to run in a 'silik' cell. And `silik` cells can only contain silik code - not python, R, ... !

To run code on sub-kernel in a silik cell, you can use `run "code" path_to_kernel` command.

The following code :

```bash
start python3 --label k1
> k1.py
1+1
```

will stop at the second line; and not execute the last line.

> You can not mix languages in a single cell

#### In `command` mode :

When you are in command mode, you can use TAB completion to display all commands and values for the arguments. Send `help` to silik kernel, or see [here](#help) for the list of commands.

#### In `connect` mode

In connect mode, silik kernel acts as a gateway to the kernel selected on `command` mode.

**Code execution** : code from the cell is sent to the kernel through its shell channel, using [Jupyter MultiKernelManager](https://jupyter-client.readthedocs.io/en/stable/api/jupyter_client.html#jupyter_client.multikernelmanager.MultiKernelManager). Output is retrieved from the iopub channel, and sent to the front-end of silik-kernel. This comprises error messages, stream, display_data, execute_result, ... See [jupyter_client documentation](https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-xpub-sub-channel).

**Code completion** : TAB completion of the selected kernel is also connected to the silik frontend;

> The only code which is not sent to sub kernel is `/cmd`. This is the exit of connect mode.

## Recursive

You can start a silik kernel from a silik kernel. But you can only control the children-silik with `run "code" path_to_sub_silik`; and not directly /cmd (because it is catched before by the first silik).

> You can hence implement your own sub-class of silik kernel, and add any method for spreading silik input to sub-kernels, and merging output of sub-kernels to produce silik output.

## Similar projects

Existing projects involving multi kernel management already exists :

- [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) : a server that is accessible through the MCP protocol, to manage multiple kernels and notebooks. To our knowledge, the MCP server does not interact with a Jupyter Kernel, but directly manages Kernels.

- [SoS Polyglot Notebook](https://vatlab.github.io/sos-docs/) : an other multi-kernel manager, through jupyter notebook. Uses a 'Super Kernel' to manage all sub-kernels. To our knowledge, the Super Kernel is not a Jupyter Kernel.

- [jupyter-kernel-mcp](https://github.com/codewithcheese/jupyter-kernel-mcp): MCP server that allows to manage multi kernels. Unlike [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server), it interact directly with jupyter kernels.

- [jupyter-code-executor-mcp-server](https://github.com/twn39/jupyter-code-executor-mcp-server) : MCP server that allows to manage multi-kernels. Deals with notebooks and not with kernels directly.

The difference between these projects and silik-kernel is the fact that we wrapped the Kernel Manager itself in a Jupyter Kernel. Moreover, the interaction with sub-kernels is not necessarly made by LLMs through an MCP server (as in [jupyter-kernel-mcp](https://github.com/codewithcheese/jupyter-kernel-mcp)) - kernels can be managed by humans first, and LLMs after :-).

Both [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) and [SoS Polyglot Notebook](https://vatlab.github.io/sos-docs/) interacts with notebooks. We propose here an interaction at a lower level : directly with the jupyter-kernel. Instead of using high-level commands to manage kernels (like [SoS Polyglot Notebook](https://vatlab.github.io/sos-docs/)), we use a lightweight bash-like language. This allows to reuse existing jupyter messaging protocol for multi-kernel management (and hence branching any front-end to it). In [SoS](https://vatlab.github.io/sos-docs/), the interaction between kernels is dealt with a protocol that allows to share variables, files, ...

> We are betting here to use 'text-only' interactions : fewer features but deployment is easier. This is possible thanks to LLM and Agent.

## Help

See [help.md](help.md) !
