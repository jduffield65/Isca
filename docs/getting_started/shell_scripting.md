# Shell Scripting
A shell script (`.sh`) can be used to submit a sequence of commands to terminal.

## Video
The following video outlines the basics of shell scripting. 

<iframe width="420" height="315"
  src="https://www.youtube.com/embed/7zJUceJiYxQ">
</iframe>

*08:32*: The first line of a shell script must be the *shebang* line, telling system which shell interpreter to use. The 
default shell interpreter can be found  by running `echo $SHELL` in terminal. This will return something like
`/bin/bash`.

*10:36*: Before running a shell script, you may need to give it execution permissions (this is similar
to the *CONDA* python [issue](kennedy.md#create-environment) with kennedy).

*13:00*: A shell script can accept any number of parameters.

## Example Script

Let's create an example script which accepts 3 parameters and prints them, as well as the file name,
called *example.sh* (execution permission can be given through `chmod u+x example.sh`):

```bash
#!/bin/bash

#This program accepts 3 parameters and prints them
echo File Name: $0
echo Param 1  : $1
echo Param 2  : $2
echo Param 3  : $3
```

Running this script through `./example.sh p1 p2 p3` prints:
```
Exec Name: ./example.sh
Param 1  : p1
Param 2  : p2
Param 3  : p3
```

## Useful Commands  
- `echo` - Use to display the value of a variable e.g. `echo $SHELL` will display the value of `SHELL`. 
- `export var=5` - this will mean that an environmental variable called `var` will be created and given the value `5`.
It can then be accessed in terminal e.g. it can be printed through `echo $var`. 
- `printenv` - This displays all the environmental variables e.g. if the parameter `SHELL` shows up in the list, 
then its value can be accessed through `$SHELL` 
- `source shell_script.sh` - This will make all variables defined through `export` in the shell script *shell_script.sh* 
be available in the current terminal. 
