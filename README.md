# Silik Kernel

> A Jupyter Multi Kernel Manager, wrapped in a Jupyter Kernel 🙂

![](example.gif)

This is a jupyter kernel that allows to interface with multiple kernels, you can:

- start, stop and restart kernels,

- switch between kernels,

- list available kernels,

- connect to a living kernel.

A Silik Kernel can be in two modes :

- **`command`** mode : manage kernels (start, stop, ...),

- **`connect`** mode : connects to one kernel, and acts as a gateway with this kernel. Support for TAB completion, and propagation of all sockets is implemented here.

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

> You can list the available kernels by running `jupyter kernelspec list` in a terminal. Or with TAB completion of `mkdir` command in a cell of a silik kernel.

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
mkdir python3 --label py
cd py
run "x = 19"
run "print(x)"
```

For persistent connection :

```bash
/cnct
```

> Controls starting with a `/` must not be mixed in the same code cell with other code (they are not the same language). See [below](#usage-guide). Always run `/cnct` in a single cell (same for `/cmd`).

```bash
print(x)
```

### Usage Guide

To switch between the two modes ('connect' and 'command'), you have to send either `/cnct` or '/cmd'.

> ‼️ Since `/cmd` and `/cnct` are not commands, they can not be run in multiline cells with command belows - neither with any other language : python, ... ‼️ To run code on sub-kernel in a silik cell, you can use `run "code"` command.

The following code :

```bash
mkdir python3 --label py
cd py
/cnct
```

will display an error; because we are mixing commands (mkdir, cd, ...) and controls (/cnct, /cmd).

#### In `command` mode :

When you are in command mode, you can use TAB completion to display all commands and values for the arguments. Here is a quick list of available commands :

• cd path : Moves the selected kernel in the kernel tree

• ls : Displays the kernels tree

• mkdir kernel_type --label=kernel_label : starts a kernel (see 'kernels' command)

• run "code" : run code on selected kernel - in one shot

• restart : restart the selected kernel

• history : displays the cells input history for this kernel

• kernels : displays the list of available kernels types

#### In `connect` mode

In connect mode, silik kernel acts as a gateway to the kernel selected on `command` mode.

**Code execution** : code from the cell is sent to the kernel through its shell channel, using [Jupyter MultiKernelManager](https://jupyter-client.readthedocs.io/en/stable/api/jupyter_client.html#jupyter_client.multikernelmanager.MultiKernelManager). Output is retrieved from the iopub channel, and sent to the front-end of silik-kernel. This comprises error messages, stream, display_data, execute_result, ... See [jupyter_client documentation](https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-xpub-sub-channel).

**Code completion** : TAB completion of the selected kernel is also connected to the silik frontend;

> The only code which is not sent to sub kernel is `/cmd`. This is the exit of connect mode.

## Recursive

You can start a silik kernel from a silik kernel. But you can only control the children-silik with `run "code"`; and not directly /cmd or /cnct (because these two are catched before by the first silik). Here is an example :

![](https://github.com/mariusgarenaux/silik-kernel/blob/main/silik_console_2.png?raw=true)

> You can hence implement your own sub-class of silik kernel, and add any method for spreading silik input to sub-kernels, and merging output of sub-kernels to produce silik output.

## Similar projects

Existing projects involving multi kernel management already exists :

- [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) : a server that is accessible through the MCP protocol, to manage multiple kernels and notebooks. To our knowledge, the MCP server does not interact with a Jupyter Kernel, but directly manages Kernels.

- [SoS Polyglot Notebook](https://vatlab.github.io/sos-docs/) : an other multi-kernel manager, through jupyter notebook. Uses a 'Super Kernel' to manage all sub-kernels. To our knowledge, the Super Kernel is not a Jupyter Kernel.

The difference between these projects and silik-kernel is the fact that we wrapped the Kernel Manager itself in a Jupyter Kernel.

Instead of using high-level commands to manage kernels (like SoS notebooks), we use a lightweight bash-like language. This allows to reuse existing jupyter messaging protocol for multi-kernel management (and hence branching any front-end to it). In SoS, the interaction between kernels is dealt with a protocol that allows to share variables, files, ... We are betting to use 'text-only' interactions : fewer features but deployment is easier. This is possible thanks to LLM and Agent.
