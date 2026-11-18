#!/bin/bash
#SBATCH --job-name=3fish_stage_midlines   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (worker processes)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (one task ~2 GB in total)
#SBATCH --time=01:00:59          # total run time limit (HH:MM:SS); ~20k EOD frames, ~0.03 s per frame per core: ~5-10 min
#SBATCH --array=0-109            # 11 recordings x 10 mask stages (python cluster_3fish_stage_midlines.py --list)
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu

# Step 1 of 3 (cluster_3fish_common.py). Submit from the repository root:   sbatch cluster_3fish_stage_midlines.slurm
# Recordings without an EOD table are skipped (FRAMES=eod); every video frame instead:   FRAMES=all sbatch ...
# Only the recording with an EOD table so far (2024-10-15T14_12_25):   sbatch --array=60-69 cluster_3fish_stage_midlines.slurm

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate electric_fish     # python 3.11, numpy >= 2.4, pandas, scipy (cluster_environment.yml)

export OMP_NUM_THREADS=1
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_3fish_stage_midlines.py
