import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import os
import numpy as np
import scipy
import scipy.stats as stats
import sys
import getopt
import subprocess
import os.path
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import random
import subprocess
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
import pylab
from utils import file_operations
from utils import model_processing

#Calculate randomized log-likelihood and save to disk
#we want to show that models dont show difference if random binding nonbinding transcripts in models
def calc_ll_ratios(data_path, log_random_path, random_path, path, disease, approach, is_sge, sampling, nr_i_bef,nr_iterations):
    jobid = None
    if not os.path.exists(random_path/'nb_t_models'/f'{nr_iterations-1}_models.pickle'):
        if os.path.exists(random_path/'nb_t_models'/f'{nr_i_bef}_models.pickle'):
            logger.error(f'Error: {nr_i_bef}_models.pickle already existed before. Stopping execution.')
            sys.exit(2)
        filtered_gene_mirna_pairs = pd.read_pickle(data_path/f'filtered_gene_mirna_pairs2.pickle')[['miRNA','ensembl_gene_id','both','nbs']]
        if not os.path.exists(random_path/'pairs'/f'gene_mirna_pairs_{nr_iterations-1}.pickle'):
            if os.path.exists(random_path/'pairs'/f'gene_mirna_pairs_{nr_i_bef}.pickle'):
                logger.error(f'Error: gene_mirna_pairs_{nr_i_bef}.pickle already existed before. Stopping execution.')
                sys.exit(2)
            logger.info('Random sample nonbinding transcripts.')
            for i in range(nr_i_bef,nr_iterations):
                if sampling: 
                    nr_sample_pairs = 10000
                    random_pairs = filtered_gene_mirna_pairs.sample(n = nr_sample_pairs).copy()
                else: 
                    random_pairs = filtered_gene_mirna_pairs.copy()
                random_pairs['nbs'] = random_pairs.apply(lambda x: random.sample(x.both, len(x.nbs)),axis=1)
                random_pairs.to_pickle(random_path/'pairs'/f'gene_mirna_pairs_{i}.pickle')
        if is_sge:
            output = subprocess.run(f'qsub -b y -t {nr_i_bef+1}-{nr_iterations+1} -o {log_random_path} -e {log_random_path} {path}/lin_reg_rndm_sge.sh {approach} {disease} {random_path}', capture_output=True, shell=True, universal_newlines=True)
            logger.info(output.stdout)
            if output.stderr != '': logger.error(output.stderr)
            jobid = output.stdout.split(' ')[2].split('.')[0]
        else:
            #execute slurm linear regression
            nr_par = 20 if sampling else 10
            process = subprocess.Popen(['sbatch',f'--array={nr_i_bef}-{nr_iterations}%{nr_par}',f'--output={log_random_path}/%a.out',f'--error={log_random_path}/%a.err',f'{path}/lin_reg_rndm_slurm.sh',f'{approach}',f'{disease}',f'{random_path}'], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            logger.debug(process.stdout.read())            

    ll_random = []
    #read in already trained full models
    all_t_models = model_processing.filter_models_by_rmse(data_path,f'all_t_models')
    logger.info(f'Waiting for first linear regression job to finish.')
    for i in range(nr_i_bef, nr_iterations):
        file = random_path/'ll'/f'log_likelihood_{i}.pickle'
        if not os.path.exists(file):
            error_file = (f'{log_random_path}/randomize_sge.sh.e{jobid}.{i+1}' if jobid!=None else None) if is_sge else f'{log_random_path}/{i}.err'
            file_operations.check_whether_exists(random_path/'nb_t_models'/f'{i}_models.pickle',90,error_file=error_file)
            #when reduced models pickle files saved, read them in 
            logger.debug(f'read in model {i}')
            nb_t_models = model_processing.filter_models_by_rmse(random_path,f'nb_t_models',prefix=f'{i}_')
            #common pairs
            common_pairs = model_processing.get_common_models(nb_t_models, all_t_models,prefix=f'{i}_')
            #log likelihood
            log_lls = model_processing.calc_ll(random_path/'ll',common_pairs,suffix=f'_{i}')
        else:
            logger.info(f'Read in log-likelihood for each (full_model,reduced_model)-pair for {i}.')
            log_lls = pd.read_pickle(file)
        ll_random.append([i,len(log_lls),len(log_lls[log_lls.corr_p_val < 0.05])])
    ll_random = pd.DataFrame(ll_random,columns=['iteration','nr_models','nr_sig_models'])
    ll_random['ratio_significant'] = ll_random.nr_sig_models/ll_random.nr_models
    ll_random.to_pickle(random_path/'random_ratios.pickle')
    logger.info(f'Finished {nr_iterations} randomized {disease} {approach} log likelihood ratio tests.')
    return ll_random

#plot ll distribution and calc prob
def significance_test(data_path, random_path, ll_random, sampling, ll_subset=None):
    #check if normal distribution, QQ-plot
    sm.qqplot(ll_random.ratio_significant.values, line='45')
    pylab.savefig(random_path/'plots'/f'ratio_significant_QQ.png')
    pylab.close() 
    real_log_lls = pd.read_pickle(data_path/f'log_likelihood.pickle')
    real_ratio = len(real_log_lls[real_log_lls.corr_p_val < 0.05])/len(real_log_lls) #percentage
    if sampling:
        try:
            fig = plt.figure(1, (7, 4))
            ax = fig.add_subplot(1, 1, 1)
            height,_,_=plt.hist(ll_random.ratio_significant,label='random shuffled transcript subsets',alpha=0.7,bins=20)
            plt.hist(ll_subset.ratio_significant,label='non-shuffled subsets',alpha=0.7,bins=20)
            plt.ylabel("Number of runs")
            plt.xlabel("Share of (miRNA, gene) models with significant likelihood ratio")
            plt.axvline(x = real_ratio, color = 'C1', linestyle="--",label='ratio complete data') 
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
            plt.savefig(random_path/'plots'/f'ratio_significant_hist.png',bbox_inches = "tight",dpi=200)
            plt.xlim([0, 1])
            plt.savefig(random_path/'plots'/f'ratio_significant_hist_far.png',bbox_inches = "tight",dpi=200)
            plt.close()
        except:
            logger.error('ll_subset has to be supplied as argument if sampling')
    else:
        fig = plt.figure(1, (7, 4))
        ax = fig.add_subplot(1, 1, 1)
        ll_random.ratio_significant.hist()
        plt.ylabel("Number of runs")
        plt.xlabel("Share of (miRNA, gene) models with significant likelihood ratio")
        plt.axvline(x = real_ratio, color = 'black', linestyle="--",label='our ratio') 
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig(random_path/'plots'/f'ratio_significant_hist.png',bbox_inches = "tight",dpi=200)
        plt.xlim([0, 1])
        plt.savefig(random_path/'plots'/f'ratio_significant_hist_far.png',bbox_inches = "tight",dpi=200)
        plt.close()
        #perform one sample t-test
        statistic, pvalue = stats.ttest_1samp(a=ll_random.ratio_significant.values, popmean=real_ratio,alternative='less')
        logger.info(f'The mean ratio is{"" if pvalue<0.05 else " not"} significantly smaller than {real_ratio} (p={pvalue}).')

#calculate log likelihood ratio for random subsets of the data and save to disk
def subset_normal_regression_ll_ratio(data_path, random_path, nr_iterations, nr_sample_pairs = 10000):
    ll_subset = []
    logger.info (f'Read in filtered (gene,miRNA)-pairs to calculate log-likelihood ratios for subsets of {nr_sample_pairs} pairs.')
    log_lls = pd.read_pickle(data_path/f'log_likelihood.pickle')
    filtered_gene_mirna_pairs = pd.read_pickle(data_path/'filtered_gene_mirna_pairs2.pickle')[['miRNA','ensembl_gene_id']].drop_duplicates().rename(columns={'ensembl_gene_id':'gene'})
    for i in range(nr_iterations):
        subset = pd.merge(log_lls,filtered_gene_mirna_pairs.sample(n = nr_sample_pairs),on=['miRNA','gene'],how='inner')
        ll_subset.append([i,len(subset),len(subset[subset.corr_p_val < 0.05])])
    ll_subset = pd.DataFrame(ll_subset,columns=['iteration','nr_models','nr_sig_models'])
    ll_subset['ratio_significant'] = ll_subset.nr_sig_models/ll_subset.nr_models
    ll_subset.to_pickle(random_path/'subset_ratios.pickle')
    return ll_subset

def main(argv):
    # parsing parameters
    path = os.path.dirname(os.path.realpath(__file__))
    logging.basicConfig(filename=f'{path}/fallback.log', filemode='a', force=True, format='%(asctime)s %(levelname)s - %(message)s')
    sampling = False
    workload_manager = None
    nr_i_bef = None
    disease = None
    approach = None
    nr_iterations = None
    try:
        opts, args = getopt.getopt(argv, 'd:a:w:i:b:s')
    except getopt.GetoptError:
        logger.error('Error: Incorrect argument. Usage: python nested_random.py -d <disease> -a <approach> -w <workload_manager> -i <nr_iterations> [-b <iterations_before> -s]')
        sys.exit(2)
    for i in opts:
        if i[0] == '-d':
            disease = i[1]
        if i[0] == '-a':
            approach = i[1]
        if i[0] == '-w':
            workload_manager = i[1]
        if i[0] == '-i':
            nr_iterations = int(i[1])
        if i[0] == '-b':
            nr_i_bef = int(i[1])
        if i[0] == '-s':
            sampling = True
    if ((disease == None)|(approach == None)|(workload_manager == None)|(nr_iterations == None)):
        logger.error('Error: Missing argument. Usage: python nested_random.py -d <disease> -a <approach> -w <workload_manager> -i <nr_iterations>')
        sys.exit(2)
    if not approach in ['A','B','C']:
        logger.error('Error: Wrong argument. -a <approach> has to be in [A,B,C]')
        sys.exit(2)
    if not workload_manager in ['sge','slurm']:
        logger.error('Error: Wrong argument. -w <workload_manager> has to be sge or slurm')
        sys.exit(2)
    else: 
        is_sge = (workload_manager == 'sge')
    if nr_i_bef == None:
        nr_i_bef = 0
        logger.info('No number of previous iterations was provided. Therfor start at 0.')
    if sampling == False:
        logger.info("No sampling preference was provided. Therfor we don't sample but instead randomize all miRNA-gene pairs.")

    overwrite = False
    nested_path = Path(path/disease)
    data_path = Path(nested_path/approach)
    if not os.path.exists(data_path):
        logger.error('Before running nested_random.py you have to run nested.py for the same disease and approach.')
        exit()
    random_path = data_path/'randomized_10000' if sampling else data_path/'randomized_all'
    if not os.path.exists(random_path):
        os.makedirs(random_path)
        os.makedirs(random_path/'plots')
        os.makedirs(random_path/'pairs')
        os.makedirs(random_path/'ll')
    log_random_path = Path(path/'res_random')
    if not os.path.exists(log_random_path):
        os.makedirs(log_random_path)

    if os.path.exists(f'{random_path}/nested_random{""if sampling else "_full"}_{disease}_{approach}.log'):
        logging.basicConfig(filename=f'{random_path}/nested_random_{disease}_{approach}.log', filemode='a', force=True, format='%(asctime)s %(levelname)s - %(message)s')
    else:
        logging.basicConfig(filename=f'{random_path}/nested_random_{disease}_{approach}.log', filemode='w', force=True, format='%(asctime)s %(levelname)s - %(message)s')
    #check before overwriting
    file = random_path/'random_ratios.pickle'
    if os.path.exists(file) and not overwrite:
        logger.info('Read in random ratios from file.')
        ll_random = pd.read_pickle(file)
    else:        
        ll_random = calc_ll_ratios(data_path, log_random_path, random_path, path, disease, approach, is_sge, sampling = sampling, nr_i_bef = nr_i_bef, nr_iterations = nr_iterations)
    if sampling: 
        file = random_path/'subset_ratios.pickle'
        if os.path.exists(file) and not overwrite:
            logger.info('Read in subset ratios from file.')
            ll_subset = pd.read_pickle(file)
        else:
            ll_subset = subset_normal_regression_ll_ratio(data_path, random_path, nr_iterations)
        significance_test(data_path, random_path, ll_random, sampling, ll_subset)
    else:
        significance_test(data_path, random_path, ll_random, sampling)

    logger.info('Done.')

if __name__ == '__main__':
    main(sys.argv[1:])
