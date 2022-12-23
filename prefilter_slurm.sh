#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=prefilter
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
srun python prefilter.py -s $((SLURM_ARRAY_TASK_ID)) -a $1 -d $2