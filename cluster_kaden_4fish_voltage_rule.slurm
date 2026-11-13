#!/bin/bash
#SBATCH --job-name=kaden_4fish_voltage_rule     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (the script uses them all through multiprocessing)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (one task peaks at ~3-4 GB in total)
#SBATCH --time=03:00:59          # total run time limit (HH:MM:SS); ~8-12k frames per task, ~0.4 s per frame per core
#SBATCH --array=0-283            # 71 videos x 4 chunks (python cluster_kaden_4fish_voltage_rule.py --list)
#SBATCH --output=slurm_logs/%x_%A_%a.out       # the folder must exist before sbatch: mkdir -p slurm_logs
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu   # from your template: change if you want mail elsewhere
##SBATCH --partition=<name>      # if Axon needs a partition (or --account) for CPU jobs, set it here

# Submit from the repository root:   mkdir -p slurm_logs && sbatch cluster_kaden_4fish_voltage_rule.slurm
# Check the inputs first:            python cluster_kaden_4fish_voltage_rule.py --check
# Every task recomputes and replaces its outputs. To resume an interrupted array without redoing finished tasks:
#   OVERWRITE=0 sbatch cluster_kaden_4fish_voltage_rule.slurm

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

# conda: batch jobs do not read your interactive shell setup, so load conda explicitly
source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate electric_fish     # needs python 3.11, numpy >= 2.4, pandas, scipy, matplotlib (cluster_environment.yml)

export OMP_NUM_THREADS=1         # one thread per worker process (the sparse solves are single-threaded)
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_kaden_4fish_voltage_rule.py
