#!/bin/bash
#SBATCH --job-name=kaden_4fish_midlines     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (the script uses them all through multiprocessing)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (one task peaks at ~2 GB in total)
#SBATCH --time=01:30:59          # total run time limit (HH:MM:SS); ~60k frames per video, ~0.07 s per frame per core: ~20-30 min
#SBATCH --array=0-70             # one task per video (python cluster_kaden_4fish_midlines.py --list)
#SBATCH --output=slurm_logs/%x_%A_%a.out       # slurm_logs/ is in the repo; without it every task fails at once with no log
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu   # from your template: change if you want mail elsewhere

# Submit from the repository root:   sbatch cluster_kaden_4fish_midlines.slurm
# Check the inputs first:            python cluster_kaden_4fish_midlines.py --check
# Only the frames with an EOD:       FRAMES=eod sbatch cluster_kaden_4fish_midlines.slurm   (default: every video frame)
# Resume without redoing finished videos:   OVERWRITE=0 sbatch cluster_kaden_4fish_midlines.slurm

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

# conda: batch jobs do not read your interactive shell setup, so load conda explicitly
source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate electric_fish     # needs python 3.11, numpy >= 2.4, pandas, scipy (cluster_environment.yml)

export OMP_NUM_THREADS=1         # one thread per worker process
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_kaden_4fish_midlines.py
