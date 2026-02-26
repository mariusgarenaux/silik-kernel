# Silik command documentation

• kernels : Returns the list of available kernel that can be started from silik.

        Example :
        ---
            In [1]: kernels
            Out[1]: ['python3', 'pydantic_ai', 'octave', 'silik']

• start : Starts a new kernel, from the root of the selected dir. Use tab completion or send 'kernels' command to see the list of available kernels.

        Positional arguments :
        ---
            • kernel_type (str) : the type of the kernel which will be started. Must
                be one of available kernels (see `kernels` command)

        Flags :
        ---
            • label (--label, -l) (str) : the label of the started kernel. A random
                label is chosen if not given. The name of the kernel is the label
                followed by the file extension of the kernel.

        Examples :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: start python3 -l k2
            Out[2]:
            ├─ k1.py
            ╰─ k2.py


            In [3]: start bash
            Out[3]:
            ├─ k1.py
            ├─ k2.py
            ╰─ lune.sh

• restart : Restart a kernel.

        Positional arguments :
        ---
            • path (str) : the path towards the kernel that will be restarted

        Example :
        ---
            In [1]: start python3 -l k1
            Out[1]:
            ╰─ k1.py


            In [2]: run "x=18" k1.py
            Out[2]:

            In [3]: run "x" k1.py
            Out[2]: 18

            In [4]: restart k1.py
            Out[4]: Restarted kernel k1.py

            In [5]: run "x" k1.py
            ---------------------------------------------------------------------------
            NameError                                 Traceback (most recent call last)
            Cell In[1], line 1
            ----> 1 x

            NameError: name 'x' is not defined

• info : Returns the path to the connection file of a kernel. The connection file stores information for frontends to connects to this kernel.

        Positional arguments :
        ---
            • path (str) : the path to the kernel to which get connection file
                path

        Example :
        ---
            In [1]: start python3 -l k1
            Out[1]:
            ╰─ k1.py

            In [2]: info k1.py
            Out[2]: /Users/username/silik/kernel-1786bfbe-3925-44bc-81e9-0fc1014d8ea6.json

• history : Display the history of the selected kernel. Sends an 'history_request' to the kernel (see https://jupyter-client.readthedocs.io/en/stable/messaging.html#history).

        > ! not all kernel return information on this message request !

        Positional arguments :
        ---
            • path (str) : the path to the kernel that will send its history

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: > k1.py
            Out[2]: All cells are executed on kernel k1.py. Run /cmd to exit this mode and select a new kernel.

            In [3]: x = 19

            In [4]: for i in range(10):
            ...:     print(i*x)
            ...:
            0
            19
            38
            57
            76
            95
            114
            133
            152
            171

            In [5]: /cmd
            Out[5]: Command mode. You can create and select kernels. Send `help` for the list of commands.

            In [6]: history k1.py
            Out[6]:
            ------- 0 -------

            x = 19

            ------- 1 -------

            for i in range(10):
                print(i*x)

• run : Send a message to the active sub kernel. Returns the result through IOPub channel.

        Positional arguments :
        ---
            • cmd (str) : the command to be sent, between quotes for complex commands
                (e.g. `run "print('hey from ipykernel')" k1.py`)

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py


            In [2]: run "x=18" k1.py
            Out[2]:

            In [3]: run "x" k1.py
            Out[2]: 18

• source : Execute the content of a text file on the silik kernel. The text file is located on the filesystem where the kernel runs. Relative paths are from where you started the jupyter kernel.

        The content must be commands that can be run on silik.
        Multiline commands are supported, but the text file must contain
        only silik commands, not code of an other language.

        Positional arguments :
        ---
            • path (str): the path (relative or absolute) towards
                 the text file

        Example :
        ---
            init.silik :
                ```silik
                start python3 --label k1
                run "x=19" k1.py
                run "x" k1.py
                ```
            In [1]: source init.txt
            Out[1]: 19

• cat : Display the content of a text file. The text file is located on the filesystem where the kernel runs. Relative paths are from where you started the jupyter kernel.

        Positional arguments :
        ---
            • path (str): the path (relative or absolute) towards the text file

        Example :
        ---
            In [3]: cat ex_scripts/ex_init.silik
            Out[3]:
            start python3 -l k1
            run "x=2" k1.py
            run "x" k1.py

• > : Changes the mode of silik-kernel to 'command'. All future code cells will be run on a sub-kernel.

        Positionals arguments :
        ---
            • path (str) : the path (relative or absolute) to the sub-kernel

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py

            In [2]: > k1.py
            Out[2]: All cells are executed on kernel k1.py. Run /cmd to exit this mode and select a new kernel.

            In [3]: 1+1
            Out[1]: 2

• gateway : Changes the mode of silik-kernel to 'command'. All future code cells will be run on a sub-kernel.

        Positionals arguments :
        ---
            • path (str) : the path (relative or absolute) to the sub-kernel

        Example :
        ---
            In [1]: start python3 --label k1
            Out[1]:
            ╰─ k1.py

            In [2]: > k1.py
            Out[2]: All cells are executed on kernel k1.py. Run /cmd to exit this mode and select a new kernel.

            In [3]: 1+1
            Out[1]: 2

• tree : Display the whole tree (directories and kernels). The current directory is displayed with '<<' at its right.

        Example :
        ---
            In [2]: tree
            Out[2]:
            ├─ chatbots
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python <<
                ╰─ k1.py

• mkdir : Creates a directory inside the current directory. A directory can be used to store kernels. They are not persistent through sessions, it is just a way to organize all kernels.

        Positional arguments :
        ---
            • label (str) : the name of the directory to create

        Example :
        ---
            In [1]: mkdir python_kernels
            Out[1]:

            In [2]: cd python_kernels/
            Out[2]: ~/python_kernels/

            In [3]: start python3 -l k1
            Out[3]:
            ╰─ python_kernels <<
                ╰─ k1.py


            In [4]: start python3 -l k2
            Out[4]:
            ╰─ python_kernels <<
                ├─ k1.py
                ╰─ k2.py

• cd : Allows to move between the folders (in silik only). Directories can be created in silik, but have no link with your real filesystem.

        It is just a way to store and organize the kernel objects.

        Positional arguments :
        ---
            • path (str): the path (relative or absolute) towards the new
                directory

        Example :
        ---
            In [2]: tree
            Out[2]:
            ├─ chatbots
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python <<
               ╰─ k1.py


            In [3]: cd ../chatbots/
            Out[3]: ~/chatbots/

            In [4]: tree
            Out[4]:
            ├─ chatbots <<
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python
               ╰─ k1.py

• ls : Displays the content of a folder (in silik kernel).

        Positional arguments :
        ---
            • path (str): the path to the directory

        Example :
        ---
            In [2]: tree
            Out[2]:
            ├─ chatbots
            │  ├─ qwen4b-dist.txt
            │  ╰─ qwen1b7-local.txt
            ╰─ python <<
               ╰─ k1.py


            In [3]: ls ../chatbots/
            Out[3]:
            ├─ qwen4b-dist.txt
            ╰─ qwen1b7-local.txt

• help : Display the help message.

        Flags :
        ---
            • cmd (--cmd) : the name of the command

        Example :
        ---
            In [1]: help --cmd kernels
            Out[1]:
            • kernels :
                    Returns the list of available kernel that can be started from silik.

                    Example :
                    ---
                        In [1]: kernels
                        Out[1]: ['python3', 'pydantic_ai', 'octave', 'silik']
