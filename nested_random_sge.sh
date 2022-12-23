#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=200G
#$ -j y
#$ -m eas
python3 nested_random.py -d "$1" -a "$2" -w "sge" -i $3 -s
