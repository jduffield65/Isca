# Pycharm
My preferred IDE for *python* is [*Pycharm*](https://www.jetbrains.com/pycharm/). The following gives instructions
on how files can be synced between your local computer and a high performance computer e.g. [kennedy](kennedy.md) 
using *Pycharm*.

Then, you basically code as you normally would on your local computer but everything happens on the high performance
computer.

## Syncing a Pycharm Project
* First, [login to kennedy](kennedy.md#login) and [create a *CONDA* environment](kennedy.md#create-environment). I will
be using an environment called `test_env`, to coincide with the
[Isca](../Isca/getting_started.md#create-conda-environment-step-3) usage.

* Next, create a *Pycharm* project. I wouldn't worry too much about the python interpreter as we will later 
change this to use the *CONDA* environment we just set up on the high performance computer. I would probably just use 
set up a *CONDA* environment or use one that is already been set up.

* Create a couple of files in the project so we can sync it to the high performance computer e.g.</br>
![image.png](../images/hpc_basics/pycharm/project.png){width="700"}

* Next choose the remote *CONDA* environment as the python interpreter by following the 5 steps indicated below.

    === "1"
        ![image.png](../images/hpc_basics/pycharm/python_interpreter.png){width="700"}
    === "2"
        ![image.png](../images/hpc_basics/pycharm/username.png){width="400"}
    === "3"
        ![image.png](../images/hpc_basics/pycharm/password.png){width="400"}
    === "4"
        ![image.png](../images/hpc_basics/pycharm/confirm.png){width="400"}
    === "5"
        ![image.png](../images/hpc_basics/pycharm/remote_conda.png){width="700"}

      **1.**Click on your current python interpreter in the bottom right (*Python 3.9 (Isca)* for me). 
      Then add a new *SSH* interpreter.</br>
      **2.**Enter your *kennedy* login details.</br>
      **3.**Enter the corresponding password.</br>
      **4.**You should get a confirmation message indicating that you connected successfully.</br>
      **5.**In the next screen, select an existing environment.</br>In the *Interpreter* section, select the python file from 
      the *CONDA* environment you want to use.</br>In the *Sync folders* section, enter the address on *kennedy* where you 
      would like the project to be saved (The project name must be the same as it is on your local computer though 
      i.e. *pythonProject* for me).

* After confirming this, you should get a *File Transfer* tab in the bottom toolbar, indicating the files 
in the project have been transferred. When changes are made to these files locally, they will also be changed 
remotely.</br>
![image.png](../images/hpc_basics/pycharm/file_transfer.png){width="700"}

    ??? warning "Remote Files don't get deleted when deleted locally"
        By default, when you delete a file locally, the remote equivalent does not get deleted. </br>
        To change this, go to Tools → Deployment → Options. </br>
        Then tick the *Delete remote files when local are deleted* box:
        === "Tools → Deployment → Options"
            ![image.png](../images/hpc_basics/pycharm/options1.png){width="400"}
        === "Delete remote files when local are deleted"
            ![image.png](../images/hpc_basics/pycharm/options2.png){width="250"}

## Python Console and Terminal
The *Python Console* in the bottom toolbar should now be using the remote *CONDA* version of python as indicated
by the first line in blue and the current path is the remote project as indicated by the last blue line, starting
`sys.path.extend`:
![image.png](../images/hpc_basics/pycharm/console.png){width="400"}

However, this *CONDA* version of python may not have anything installed yet, hence I get the error when trying 
to import `numpy`. To install a package on the remote *CONDA*, go to the *Terminal* tab in the bottom toolbar and 
start a *SSH* terminal session by clicking the downward arrow and selecting the correct *Remote Python* option:

![image.png](../images/hpc_basics/pycharm/terminal.png){width="500"}

This will log you into *kennedy* without asking for login details seen as you have already provided them.
Then, [activate](kennedy.md#create-environment) the *CONDA* environment and install the package (`pip install numpy`).
If you then restart the *Python Console*, you should now be able to import `numpy`.

## Debugging
I think that one of the main advantages of using *Pycharm* is the debugging feature, so you can pause a function 
in real time to see the variables or see why it is hitting an error. This feature can also be used in this remote 
setup.

First, create a [run configuration](https://www.jetbrains.com/help/pycharm/run-debug-configuration.html) for the 
script that you want to run. I will be using `script1.py`:

![image.png](../images/hpc_basics/pycharm/script.png){width="500"}

Then, to run in debug mode, click the little beetle in the top right. If you then add a breakpoint somewhere, 
the code should stop at that point, so you can see the value of all the variables:

![image.png](../images/hpc_basics/pycharm/debugging.png){width="700"}

