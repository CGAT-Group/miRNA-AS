#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=lin_reg
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
srun python linear_regression.py -s $((SLURM_ARRAY_TASK_ID)) -a $1 -d $2