from isca_tools.jasmin import run_script
import os

use_slurm = True
do_local = False
if do_local:
    jobs_dir = os.path.join(os.environ['HOME'], 'Documents', 'StAndrews', 'Isca', 'jobs')
    use_slurm = False
else:
    # on JASMIN
    jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')  # all jobs saved here - CHANGE FOR EACH EXPERIMENT
# Test
# script_path = os.path.join(jobs_dir, 'jasmin/test/test_multi_args.py')
# script_args = ['Barry', 52, 1.98, ['football','tennis'], True]
# run_script(script_path, script_args, slurm=True, mem=1, partition='debug',
#            time='00:05:00', exist_output_ok=None)


# Run analysis job with an input file
# exp_input_file = os.path.join(jobs_dir, f'cesm/theory_adiabat/hottest_lat_lon/co2_0_5x/av/input{"_local" if do_local else ""}.nml')
# exp_input_file = os.path.join(jobs_dir, 'cesm/theory_adiabat/tropics_summer/co2_0_5x/ocean/input.nml')
# run_script(input_file_path=exp_input_file, slurm=use_slurm)


# # Get era5 data - from nml input file, run 1 job per year and wait for previous job to finish to run next
# # exp_input_file = os.path.join(jobs_dir, f'era5/climate_data_store/test/input{"_local" if do_local else ""}.nml')
# exp_input_file = os.path.join(jobs_dir, f'era5/zhang_2023/raw/500hPa_daily_mean/temperature/input.nml')
# from jobs.era5.climate_data_store.get_data import create_years_per_job_nml
# exp_input_file_job_list = create_years_per_job_nml(exp_input_file, exist_ok=True)   # exist_ok=True to re-create individual nml for each year
# job_id_prev = None
# for file in exp_input_file_job_list:
#     job_id_prev = run_script(input_file_path=file, slurm=use_slurm, dependent_job_id=job_id_prev if use_slurm else None)


# Process era5 data on Jasmin
# exp_input_file = os.path.join(jobs_dir, f'era5/jasmin_processing/test/input.nml')
exp_input_file = os.path.join(jobs_dir, f'era5/zhang_2023/raw/daily_mean/sp/input.nml')
years_per_job = 5
from jobs.era5.jasmin_processing.process_var_year_file import create_years_per_job_nml
exp_input_file_job_list = create_years_per_job_nml(exp_input_file, years_per_job, exist_ok=True) # exist_ok=True to re-create individual nml for each year
for file in exp_input_file_job_list:
    job_id_prev = run_script(input_file_path=file, slurm=use_slurm) # run each job simultaneously as no need for dependencies


# Get era5 data from python script - changed to using nml file like above
# script_path = os.path.join(jobs_dir, 'era5/zhang_2023/download_data/daily_average_pl.py')
# level_type = 'pressure'
# var = 'temperature'
# pl = 500    # 500hPa level
# for year in range(1979, 1980):
#     run_script(script_path=script_path, script_args=[var, year, level_type, pl], slurm=True,
#                partition='standard', time='24:00:00')
