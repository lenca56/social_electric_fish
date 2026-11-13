#!/bin/bash
#SBATCH --job-name=kaden_4fish_voltage_rule     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (the script uses them all through multiprocessing)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=03:00:59          # total run time limit (HH:MM:SS); ~10k frames per task at ~0.3 s per frame per core
#SBATCH --array=0-283            # 71 videos x N_CHUNKS = 4 chunks (python cluster_kaden_4fish_voltage_rule.py --list)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2023.3
conda activate electric_fish

export OMP_NUM_THREADS=1         # one thread per worker process (the sparse solves are single-threaded)
export PROJECT_DIR=$SLURM_SUBMIT_DIR

python cluster_kaden_4fish_voltage_rule.py
