#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=nested
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1
srun python nested.py -d "$1" -a "$2" -w "slurm"