#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=rnd_lin_reg
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1
srun python linear_regression.py -s $((SLURM_ARRAY_TASK_ID)) -a "$1" -d "$2" -r "$3"