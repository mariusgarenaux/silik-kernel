# Silik Kernel

This is a jupyter kernel that allows to interface with multiple kernels, you can:

- start, stop and restart kernels,

- switch between kernels,

- list available kernels.

As a jupyter kernel, it takes text as input, transfer it to appropriate sub-kernel; and returns the result in a cell output.

> **Any jupyter kernel can be plugged to silik**

![](https://github.com/mariusgarenaux/silik-kernel/blob/main/silik_console.png?raw=true)

> But managing multi-kernels seems to be a nightmare ?

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

> You can list the available kernels by running `jupyter kernelspec list` in a terminal.

## Usage

### Tuto

Start by running `mkdir <kernel_type> --label=my-kernel` with <kernel_type> among ['code-helper', 'python3', 'pydantic_ai', 'silik', 'rudi', 'ir', 'test_kernel'].

Then, you can run `cd my-kernel` and, `run <code>` to run one shot code in this kernel.

You can also run /cnct to avoid typing `run`. /cmd allows at any time to go back to command mode (navigation and creation of kernels).

### Commands

Here is a quick reminder of available commands

• cd <path> : Moves the selected kernel in the kernel tree

• ls | tree : Displays the kernels tree

• mkdir <kernel_type> --label=<kernel_label> : starts a kernel (see 'kernels' command)

• run `code` | r `code` : run code on selected kernel - in one shot

• restart : restart the selected kernel

• branch <kernel_label> : branch the output of selected kernel to the input of one of its children. Output of parent kernel is now output of children kernel. (In -> Parent Kernel -> Children Kernel -> Out)

• detach : detach the branch starting from the selected kernel

• history : displays the cells input history for this kernel

• kernels : displays the list of available kernels types

• /cnct : direct connection towards selected kernel : cells will be directly executed on this kernel; except if cell content is '/cmd'

• /cmd : switch to command mode (default one) - exit /cnct mode

## Recursive

You can start a silik kernel from a silik kernel. But you can only control the children-silik with 'run `code`'; and not directly /cmd or /cnct (because these two are catched before by the first silik). Here is an example :

![](https://github.com/mariusgarenaux/silik-kernel/blob/main/silik_console_2.png?raw=true)

> You can hence implement your own sub-class of silik kernel, and add any method for spreading silik input to sub-kernels, and merging output of sub-kernels to produce silik output.
