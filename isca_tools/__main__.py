# This file allows the Isca experiment to be called from the command line without
# entering the Python interpreter.  To call, use:
#
#     python3 -m isca_tools namelist.nml diag_table

from run import run_experiment
import sys
import os
import textwrap


def print_usage(message=None):
    if message:
        message = f"\n\n    ERROR: {message}"
    else:
        message = ''
    USAGE = f"""
    === Isca Tools Software ===

    To run Isca experiment, pass three arguments. 
    The first containing the namelist.nml file.
    The second containing the diagnostic table.
    The third indicates whether to run on slurm or not (if not given, assumes False). E.g.,

        python3 -m isca_tools namelist.nml diag_table True
    {message}
    """
    exit(textwrap.dedent(USAGE))


# Ensure there is exactly one argument, and it is an ini file
if len(sys.argv) == 1:
    print_usage("Please pass the namelist and diagnostic table files as arguments")
if len(sys.argv) >= 5:
    print_usage(f"Please only pass namelist file as first argument, diagnostic table file as the second "
                f"and optionally, True or False, indicating whether to run on Slurm as third.\n"
                f"But {len(sys.argv) - 1} arguments passed:\n{sys.argv[1:]}")
if sys.argv[1] in ["--help", "-h"]:
    print_usage()
for i in range(2):
    if not os.path.isfile(sys.argv[1 + i]):
        print_usage(f"Cannot find path {sys.argv[1 + i]}, please specify a valid file")

if len(sys.argv) == 3:
    run_experiment(sys.argv[1], sys.argv[2])
if len(sys.argv) == 4:
    use_slurm = True if "rue" in sys.argv[3] else False
    run_experiment(sys.argv[1], sys.argv[2], use_slurm)
