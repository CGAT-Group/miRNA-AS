#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=2G
#$ -N prefilter
#$ -m as
ulimit -c 0
let i=$SGE_TASK_ID-1
python3 prefilter.py -s $i -a "$1" -d "$2" -j $JOB_ID

