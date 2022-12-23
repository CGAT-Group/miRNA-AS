#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=rndm
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1
srun python nested_random.py -d "$1" -a "$2" -w "slurm" -i $3 -s