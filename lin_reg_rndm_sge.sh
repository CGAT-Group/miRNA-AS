#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=200G
#$ -N random_lin_reg
#$ -m as
ulimit -c 0
let i=$SGE_TASK_ID-1
python3 linear_regression.py -s $i -a "$1" -d "$2" -r "$3" -j $JOB_ID
