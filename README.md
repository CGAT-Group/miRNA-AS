# Alternative splicing impacts microRNA regulation within coding regions

A pipeline to analyse the influence of alternative splicing on miRNA regulation using TCGA transcript and miRNA expression data and TarPmiR miRNA target site prediction. This repository provides the code used for Hackl et al, 2023.

## Abstract
MicroRNAs (miRNAs) are small non-coding RNA molecules that bind to target sites in different gene regions and regulate post-transcriptional gene expression. Approximately 95% of human multi-exon genes can be spliced alternatively, which enables the production of functionally diverse transcripts and proteins from a single gene. Through alternative splicing, transcripts might loose the exon with the miRNA target site and become unresponsive to miRNA regulation. To check this hypothesis, we studied the role of miRNA target sites in both coding and noncoding regions using six cancer data sets from The Cancer Genome Atlas (TCGA). First, we predicted miRNA target sites on mRNAs from their sequence using TarPmiR. MiRNAs usually downregulate gene expression, thus we expect a negative correlation between miRNA and gene expression. To check whether alternative splicing interferes with this regulation, we trained linear regression models to predict miRNA expression from transcript expression. Using nested models, we compared the predictive power of transcripts with miRNA target sites in the coding regions to that of transcripts without target sites. Models containing transcripts with target sites perform significantly better. We conclude that alternative splicing does interfere with miRNA regulation by skipping exons with miRNA target sites within the coding region.

## Installation
Clone the repo. To easily get the necessary libraries you can create a conda environment using the provided requirements.txt. As a prerequisite conda should be installed on your system.
```bash
conda config --append channels conda-forge
conda config --append channels bioconda
conda env create -n as -f requirements.txt
conda activate as
```
The code is adaptable for all TCGA cancer types by specifying the [cancer type abbreviations](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations).
The approach can be chosen as A, B or C to focus on   
* A: the impact of target sites in coding region without any other target sites (TNBN)
* B: the impact of target sites in coding region with existing target sites in noncoding regions (TBN)
* C: the impact of any target sites independent of region (ALLT)

As the nature of working with big files demands, calculation of prefiltering and linear regression were parallelized, therefor the cluster has to be working with either 'sge' (Sun Grid Engine) or 'slurm' as workload management software.
Necessary input data can be found [here](https://doi.org/10.6084/m9.figshare.21821181.v1) as described in the paper. Before execution folder input_data has to be downloaded and placed in the same folder with the code.

## Usage
The code can be run from the command line as
```bash
python nested.py -d <disease> -a <approach> -w <workload_manager>
```
Run the randomization afterwards as 
```bash
python nested_random.py -d <disease> -a <approach> -w <workload_manager> -i <nr_iterations> [-b <iterations_before>] [-s]
```
or using the provided bash files nested_<workload_manager>.sh and nested_random_<workload_manager>.sh.

The necessary output folders and files will be produced automatically in the execution folder.

## Contact

For comments or questions please contact Lena Maria Hackl via [mail](mailto:lena.hackl@uni-hamburg.de).

