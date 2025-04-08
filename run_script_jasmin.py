from isca_tools.jasmin import run_script
import os

jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')  # all jobs saved here - CHANGE FOR EACH EXPERIMENT

# Test
script_path = os.path.join(jobs_dir, 'jasmin/test/test_multi_args.py')
script_args = ['Barry', 52, 1.98, ['football','tennis'], True]
run_script(script_path, script_args, slurm=True, mem=1, partition='debug',
           time='00:05:00', exist_output_ok=None)


# exp_input_file = os.path.join(jobs_dir, 'cesm/theory_adiabat/lat_quant/co2_2x/input.nml')
# run_script(input_file_path=exp_input_file, slurm=False)
