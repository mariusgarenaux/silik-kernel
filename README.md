# Silik Kernel

This is a jupyter kernel that allows to interface with multiple kernels, you can:

- start, stop and restart kernels,

- switch between kernels,

- list available kernels.

As a jupyter kernel, it takes text as input, transfer it to appropriate sub-kernel; and returns the result in a cell output. It gives a **single context** that is shared between kernels. The cell history is shared with sub-kernels within the 'metadata' attribute of execution messages.

> **Any kernel can be plugged to silik**

![](https://github.com/mariusgarenaux/silik-kernel/blob/main/silik_console.png?raw=true)

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

Once the kernel is started, you can :

- send commands :
  - `!help` : display this message
  - `!start <kernel_type>` : starts a kernel; it will be assigned a label. Per example, `!start python3` starts and connect to a python3 kernel.
  - `!restart <kernel_label>` : restart a kernel with its label.
  - `!ls` : list started kernels.
  - `!select <kernel_label>`: switch a started kernel with its label

- run code :
  - if you run `!ls`, you'll see which kernel you are on.
  - all cells you send will be executed in this kernel, and the result will be given in the cell output. Silik kernel acts as a gateway for the sub-kernels.

## Retrieving cells history with a custom kernel

If you want to retrieve the history of the silik kernel within your custom kernel, you just have to access the 'metadata' attribute of the current message. For example (assuming you use subclass the ipykernel, and self is your kernel instance) :

```python
parent = self.get_parent()
metadata = parent.get("metadata", {})
if isinstance(metadata, dict) and "message_history" in metadata:
    print(metadata["message_history"])
```

The attribute 'message_history' of the metadata is a list of dict, each with :

```python
{
  "role": "user or assistant; user for cell input, assistant for cell output",
  "content": "Input Code if user, output if assistant",
  "uid": "uuidv4"
}
```
