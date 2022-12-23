#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=200G
#$ -j y
#$ -m eas
ulimit -c 0
python3 nested.py -d "$1" -a "$2" -w "sge"
