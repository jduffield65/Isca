# Imperial HPC

Isca experiments can be submitted to Imperial's PBS scheduler by selecting the `pbs` scheduler:

```python
from isca_tools.run import run_experiment

run_experiment(namelist_file, diag_table_file, scheduler='pbs')
```

The runner submits one PBS job per simulation chunk and makes each job depend on the previous chunk. Set
`n_nodes`, `n_cores`, and `max_walltime` in the namelist's `experiment_details` section. `max_walltime`
should be in `HH:MM:SS` format. The PBS request uses
`select=<n_nodes>:ncpus=<n_cores>:mpiprocs=<n_cores>`. Add `memory_per_node` (for example, `'16gb'`) to
request memory per node; otherwise PBS uses its default of 1 GB per node.

Imperial uses routing queues, so the PBS scripts do not specify a queue. The Slurm-specific `partition` and
`nodelist` options are not used. For current queue limits and resource advice, see Imperial's
[job sizing guidance](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/job-sizing-guidance/).
