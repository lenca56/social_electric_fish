#!/bin/bash
#SBATCH --job-name=3fish_stage_predict   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (mine: worker processes; denis: numba threads)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (mine ~1 GB per worker; denis a few GB)
#SBATCH --time=04:00:59          # total run time limit (HH:MM:SS); mine: ~20k frames x 3 fish ~0.5-1.5 h on 4 cores; denis minutes
#SBATCH --array=0-549            # 11 recordings x 10 stages x 5 settings (python cluster_3fish_stage_predict.py --list)
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu

# Step 2 of 3 (cluster_3fish_common.py), after cluster_3fish_stage_midlines.slurm. Submit from the repository root:
#   sbatch cluster_3fish_stage_predict.slurm                      all tasks (recordings without an EOD table exit at once)
#   sbatch --array=300-349 cluster_3fish_stage_predict.slurm      only 2024-10-15T14_12_25 (10 stages x 5 settings)
#   MASK_STAGES=final sbatch --array=30-34 cluster_3fish_stage_predict.slurm   only the final masks of 14_12_25 (5 settings)
# Settings (INTERIOR_SIGMAS, INCLUDE_DENIS) are in the hyperparameters of cluster_3fish_stage_predict.py; if you change them,
# the task count changes too (--list). Denis' setting needs ../EODAssignment (or DENIS_REPO) and numba.
# Then, anywhere (no solves):   python assign_3fish_stage_rules.py

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate electric_fish     # python 3.11, numpy >= 2.4, pandas, scipy, matplotlib, numba (cluster_environment.yml)

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_3fish_stage_predict.py
