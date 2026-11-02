import os
from isca_tools.jasmin import run_script
from isca_tools.cesm.base import create_per_job_nml

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


# Get pressure level data for CESM
# for exp in ['pre_industrial', 'co2_2x', 'co2_4x', 'co2_0_5x']:
# years_per_job = 3
# for exp in ['co2_4x', 'co2_0_5x']:
#     exp_input_file = os.path.join(jobs_dir, f'cesm/pressure_levels/lcl/{exp}/input_lev.nml')
#     # run_script(input_file_path=exp_input_file, slurm=use_slurm)
#     exp_input_file_job_list = create_years_per_job_nml(exp_input_file, years_per_job, exist_ok=True) # exist_ok=True to re-create individual nml for each year
#     for file in exp_input_file_job_list:
#         run_script(input_file_path=file, slurm=use_slurm) # run each job simultaneously as no need for dependencies

# Get quant info data for CESM at REFHT and 500hPa
# for exp in ['co2_2x', 'co2_4x', 'co2_0_5x']:
# for exp in ['pre_industrial', 'co2_2x', 'co2_4x', 'co2_0_5x']:
# for exp in ['test']:
# for exp in ['pre_industrial', 'co2_2x', 'co2_4x', 'co2_0_5x']:
#     exp_input_file = os.path.join(jobs_dir, f'cesm/theory_adiabat/above_quant_2m_500hPa/{exp}/all_days/input_lev_quant95.nml')
#     run_script(input_file_path=exp_input_file, slurm=use_slurm)

# Save temp dist info for CESM at lev=-3 and 500hPa
# for exp in ['pre_industrial', 'co2_2x', 'co2_4x', 'co2_0_5x']:
#     for surf in ['land', 'ocean']:
#         exp_input_file = os.path.join(jobs_dir, f'cesm/theory_adiabat/temp_dist/6month_summer_lev/{exp}/{surf}/input.nml')
#         run_script(input_file_path=exp_input_file, slurm=use_slurm)

# Get quant info data for CESM TZ profiles
# for exp in ['pre_industrial', 'co2_2x', 'co2_4x', 'co2_0_5x']:
#     exp_input_file = os.path.join(jobs_dir, f'cesm/av_profile/above_TREFHTquant/Q/{exp}/input.nml')
#     run_script(input_file_path=exp_input_file, slurm=use_slurm)

# Get CESM time_ind for 3 hourly simulations
# exp_input_file = os.path.join(jobs_dir, f'cesm/3_hour/hottest_quant/pre_industrial/REFHT_quant50/T_Q_PS_all_lat/input_ind.nml')
# run_script(input_file_path=exp_input_file, slurm=use_slurm)

# Get CESM data at hottest time for 3 hourly simulations
# exp_input_file = os.path.join(jobs_dir, f'cesm/3_hour/hottest_quant/co2_2x/REFHT_quant50/T_Q_PS_all_lat/input.nml')
# exp_input_file_job_list = create_per_job_nml(exp_input_file, files_per_job=4, exist_ok=True)
# for i in range(len(exp_input_file_job_list)):
#     run_script(input_file_path=exp_input_file_job_list[i], slurm=use_slurm)


# # Specific script
# run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_july_hottest_day.py',
#            slurm=False)
# run_script('/home/users/jamd1/Isca/jobs/cesm/3_hour/hottest_quant/combine_file_output.py',
#            script_args='pre_industrial', slurm=True, time = '24:00:00', mem=100)
# run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_temp_ft_climatology.py',
#            script_args=['co2_2x', 400*100, 3, False], slurm=True, time = '24:00:00', mem=100)
# run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_ds_quant50_av.py',
#            slurm=True, time = '24:00:00', mem=100)


# ERA5 annual mean of variable
from jobs.era5.surface_flux.get_av_annual_cycle import create_var_per_job_nml
exp_input_file = os.path.join(jobs_dir, f'era5/surface_flux/av_annual_cycle/input_20years.nml')
exp_input_file_var_list = create_var_per_job_nml(exp_input_file, exist_ok=True)
for file_var in exp_input_file_var_list:
    run_script(input_file_path=file_var, slurm=use_slurm)


# # Get era5 data - from nml input file, run 1 job per year and wait for previous job to finish to run next
# # exp_input_file = os.path.join(jobs_dir, f'era5/climate_data_store/test/input{"_local" if do_local else ""}.nml')
# exp_input_file = os.path.join(jobs_dir, f'era5/diurnal_play/canada/input_single_lev_local.nml')
# exp_input_file = os.path.join(jobs_dir, f'era5/diurnal_play/canada/input_pressure_local.nml')
# run_script(input_file_path=exp_input_file, slurm=False)
# from jobs.era5.climate_data_store.get_data import create_years_per_job_nml, create_var_per_job_nml
# exp_input_file = os.path.join(jobs_dir, 'era5/surface_flux/example_ocean_coord/input.nml')
# exp_input_file_var_list = create_var_per_job_nml(exp_input_file, exist_ok=True)
# job_id_prev = None
# for file_var in exp_input_file_var_list:
#     exp_input_file_year_list = create_years_per_job_nml(file_var, exist_ok=True)   # exist_ok=True to re-create individual nml for each year
#     for file_var_year in exp_input_file_year_list:
#         job_id_prev = run_script(input_file_path=file_var_year, slurm=use_slurm,
#                                  dependent_job_id=job_id_prev if use_slurm else None)


# Process era5 data on Jasmin
# exp_input_file = os.path.join(jobs_dir, f'era5/jasmin_processing/test/input.nml')
# exp_input_file = os.path.join(jobs_dir, f'era5/daily_mean/d2m/input.nml')
# years_per_job = 1
# from jobs.era5.jasmin_processing.process_var_year_file import create_years_per_job_nml
# exp_input_file_job_list = create_years_per_job_nml(exp_input_file, years_per_job, exist_ok=True) # exist_ok=True to re-create individual nml for each year
# for file in exp_input_file_job_list:
#     job_id_prev = run_script(input_file_path=file, slurm=use_slurm) # run each job simultaneously as no need for dependencies


# Get era5 data from python script - changed to using nml file like above
# script_path = os.path.join(jobs_dir, 'era5/zhang_2023/download_data/daily_average_pl.py')
# level_type = 'pressure'
# var = 'temperature'
# pl = 500    # 500hPa level
# for year in range(1979, 1980):
#     run_script(script_path=script_path, script_args=[var, year, level_type, pl], slurm=True,
#                partition='standard', time='24:00:00')
