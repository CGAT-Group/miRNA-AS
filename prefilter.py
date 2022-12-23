import pandas as pd
from pathlib import Path
import pickle
import sys
import getopt
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# read in filtered exon and miRNA expression from file
def read_in_filtered_expression(data_path):
    logger.info('read in filtered gene and miRNA expression from files')
    gene_counts = pd.read_pickle(data_path/'filtered_gene_counts.pickle')
    mirna_counts = pd.read_pickle(data_path/'filtered_miRNA_counts.pickle')
    return gene_counts, mirna_counts

# calculate correlation between gene and miRNA expression
def prefilter(mirna_expr, gene_expr, mirna, gene):
    expr = mirna_expr.merge(gene_expr, on='case_id', how='left')
    if mirna_expr.empty:
        logger.error(f'Error: mirna expression for {mirna} empty.')
    if gene_expr.empty:
        logger.error(f'Error: gene expression for {gene} empty.')
    return expr['mirna_count'].corr(expr['gene_count'])

def main(argv):
    # parsing parameters
    try:
        opts, args = getopt.getopt(argv, 's:a:d:j:p:')
    except getopt.GetoptError:
        print('Error: Incorrect argument. Usage: python prefilter.py -s <slurm_id> -a <approach> -d <disease> [-j <sge_job_id>] [-p <data_path>]')
        sys.exit(2)
    array_id = None
    approach = None
    disease = None
    job_id = None
    given_path = None
    for i in opts:
        if i[0] == '-s':
            array_id = int(i[1])
        if i[0] == '-a':
            approach = i[1]
        if i[0] == '-d':
            disease = i[1]
        if i[0] == '-j':
            job_id = i[1]
        if i[0] == '-p':
            given_path = i[1]
            if not os.path.exists(given_path):
                logger.error(f"Error: Path {given_path} doesn't exist.")
                sys.exit(2)
    if ((array_id == None)|(approach == None)|(disease == None)):
        print('Error: Missing argument. Usage: python prefilter.py -s <array_id> -a <approach> -d <disease>')
        sys.exit(2)
    else:
        path = os.path.dirname(os.path.realpath(__file__))
        is_sge = (job_id!=None)
        if given_path != None: path = given_path
        nested_path = Path(path)/disease
        data_path = Path(nested_path/approach)
        log_path = path/Path('res_prefilter')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = f'{log_path}/prefilter_sge.sh.o{job_id}.{array_id+1}' if is_sge else f'{log_path}/{array_id}.out'
        logging.basicConfig(filename=log_file, filemode='a', force=True, format='%(asctime)s %(levelname)s - %(message)s')
        gene_counts, mirna_counts = read_in_filtered_expression(data_path)
        logger.info(f'read in gene_counts and mirna_counts for {array_id}')
        with (open(Path(data_path/'split_pairs_10000'/f'gene_mirna_pairs_{array_id}.pickle'), "rb")) as fp:
            gene_mirna_pairs = pickle.load(fp)
        logger.info(f'loaded {len(gene_mirna_pairs)} gene_mirna_pairs from disk')
        corrs = [prefilter(mirna_counts[mirna_counts['sample'] == mirna], gene_counts[gene_counts['sample'] == gene], mirna, gene) for [
            mirna, gene] in gene_mirna_pairs[['miRNA', 'ensembl_gene_id']].values]
        gene_mirna_pairs['corr'] = corrs
        if not os.path.exists(data_path/'prefilter_corrs'):
            os.makedirs(data_path/'prefilter_corrs')
        gene_mirna_pairs.to_pickle(data_path/'prefilter_corrs'/f'corrs_{array_id}.pickle')

if __name__ == '__main__':
    main(sys.argv[1:])
