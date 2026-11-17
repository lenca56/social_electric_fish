#!/bin/bash
#SBATCH --job-name=kaden_4fish_assign     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (mine: worker processes; denis: numba threads)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (mine: ~1 GB per worker; denis: a few GB for a whole video)
#SBATCH --time=08:00:59          # total run time limit (HH:MM:SS); mine: a whole video (~34k frames) ~1-3 h on 4 cores; denis minutes
#SBATCH --array=0-283            # mine: 71 videos x 4 interior sigmas (1, 4, 16, 100); denis: give --array=0-70 to sbatch
#SBATCH --output=slurm_logs/%x_%A_%a.out       # slurm_logs/ is in the repo; without it every task fails at once with no log
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu   # from your template: change if you want mail elsewhere

# Run cluster_kaden_4fish_midlines.slurm first: this job reads its midlines.
# Submit from the repository root, choosing the model:
#   MODEL=mine  sbatch cluster_kaden_4fish_assign.slurm                 my numerical model (default): 284 tasks, one whole video x
#                                                                       one interior sigma each (task 4v + i: video v, sigma i)
#   MODEL=denis sbatch --array=0-70 cluster_kaden_4fish_assign.slurm    Denis' analytic pipeline: 71 tasks, one whole video each
#                                                                       (needs ../EODAssignment, or set DENIS_REPO)
# List the tasks:           MODEL=mine python cluster_kaden_4fish_assign.py --list
# Check the inputs first:   MODEL=mine python cluster_kaden_4fish_assign.py --check
# Resume without redoing finished tasks:   MODEL=mine OVERWRITE=0 sbatch cluster_kaden_4fish_assign.slurm

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

# conda: batch jobs do not read your interactive shell setup, so load conda explicitly
source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate electric_fish     # needs python 3.11, numpy >= 2.4, pandas, scipy, matplotlib, numba (cluster_environment.yml)

export MODEL=${MODEL:-mine}      # mine or denis (from the sbatch command line; default mine)
export OMP_NUM_THREADS=1         # one thread per worker process (the sparse solves are single-threaded)
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Denis' pipeline: numba threads = the task's cores
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_kaden_4fish_assign.py
