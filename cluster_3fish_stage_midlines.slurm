#!/bin/bash
#SBATCH --job-name=3fish_stage_midlines   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (worker processes)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (one task ~2 GB in total)
#SBATCH --time=01:00:59          # total run time limit (HH:MM:SS); ~20k EOD frames, ~0.03 s per frame per core: ~5-10 min
#SBATCH --array=0-79             # 8 recordings x 10 mask stages; with MASK_STAGES=final give --array=0-7 (--list)
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu

# Step 1 of 3 (cluster_3fish_common.py). Submit from the repository root:   sbatch cluster_3fish_stage_midlines.slurm
# The 8 recordings with an EOD table (cluster_3fish_common.VIDEOS), only frames with an EOD (FRAMES=eod; FRAMES=all: every frame).
#   MASK_STAGES=final sbatch --array=0-7 cluster_3fish_stage_midlines.slurm      final masks only: 8 tasks, one per recording
#   sbatch cluster_3fish_stage_midlines.slurm                                    all 10 mask stages: 80 tasks

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate electric_fish     # python 3.11, numpy >= 2.4, pandas, scipy (cluster_environment.yml)

export OMP_NUM_THREADS=1
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_3fish_stage_midlines.py
