#!/home/hackl/miniconda3/bin/python3
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import pickle
from sklearn.metrics import mean_squared_error
import sys, os
import getopt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#save model state and input train and test data to disk as pickle files
def save_model_state(path, models, train_data, test_data, prefix):
    Path(path).mkdir(exist_ok=True)
    with open((path/(prefix+'_models.pickle')),"wb") as file:
        pickle.dump(models,file, protocol=pickle.HIGHEST_PROTOCOL)
    with open((path/(prefix+'_test_data.pickle')),"wb") as file:
        pickle.dump(test_data,file, protocol=pickle.HIGHEST_PROTOCOL)
    with open((path/(prefix+'_train_data.pickle')),"wb") as file:
        pickle.dump(train_data,file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Trained miRNA-gene-level models were saved to disk.')

#split data into train and test subsets with trainsplit between 0 and 1
def split_train_test(data, train_split=0.8):
    train_split = int(train_split * len(data))
    train = data[:train_split]
    test = data[train_split:]
    return train, test

#train linear regression model between miRNA expression and tramscript expression for provided miRNA-gene pair
def train_lin_reg_transcript(mirna, gene, transcripts, i, mirna_counts, transcript_counts, models, train_data, test_data):
    old_transcripts = transcripts.copy()
    transcripts = [c for c in transcripts if c in transcript_counts.columns]
    if len(transcripts) < len(old_transcripts):
        logger.error(f"We don't have expression data for transcripts {[c for c in old_transcripts if c not in transcript_counts.columns]} for gene {gene}")
    columns_ = transcripts.copy()
    columns_.append('case_id')
    transcript_counts_gene = transcript_counts[columns_]
    mirna_counts_mirna = mirna_counts[mirna_counts['miRNA_id'] == mirna]
    info = mirna_counts_mirna.merge(transcript_counts_gene,on='case_id',how='left')
    y = info.mirna_count
    X = info[transcripts].values
    X = sm.add_constant(X) #Documentation: An intercept is not included by default and should be added by the user
    X_train, X_test = split_train_test(X)
    y_train, y_test = split_train_test(y)
    regr = sm.OLS(y_train,X_train).fit() #estimation by ordinary least squares
    y_pred = regr.predict(X_test)
    rms = mean_squared_error(y_test, y_pred, squared=False)
    models[(mirna,gene)] = (rms, regr, regr.summary2().tables[1]['P>|t|'].values, regr.f_pvalue)
    train_data[(mirna,gene)] = (info, X_train, y_train)
    test_data[(mirna,gene)] = (info, X_test, y_test)
    if i%1000 == 0: 
        logger.debug(f'{i} done')

#train linear regression models for all miRNA-gene pairs and save them to disk
def run_linear_regression(data_path, mirna_counts, transcript_counts, gene_mirna_pairs, array_id, random, random_path):
    if not random: #full models don't change with randomization & don't have to be retrained
        logger.info(f'Number (mirna,gene)-pairs: {len(gene_mirna_pairs)}')
        models = {}
        train_data = {}
        test_data = {}
        for i, [mirna, gene, transcripts] in enumerate(gene_mirna_pairs[['miRNA','ensembl_gene_id','both']].values):
            train_lin_reg_transcript(mirna, gene, transcripts, i, mirna_counts, transcript_counts, models, train_data, test_data)
        save_model_state(data_path/'all_t_models', models, train_data, test_data,f'{array_id}')
    models = {}
    train_data = {}
    test_data = {}
    for i, [mirna, gene, transcripts] in enumerate(gene_mirna_pairs[['miRNA','ensembl_gene_id','nbs']].values):
        train_lin_reg_transcript(mirna, gene, transcripts, i, mirna_counts, transcript_counts, models, train_data, test_data)
    save_model_state((data_path if not random else random_path)/'nb_t_models', models, train_data, test_data,f'{array_id}')

def main(argv):
    # parsing parameters
    try:
        opts, args = getopt.getopt(argv, 's:a:d:r:j:p:')
    except getopt.GetoptError:
        print('Error: Incorrect argument. Usage: python linear_regression.py -s <array_id> -a <approach> -d <disease> [-r <random_path>] [-j <sge_job_id>] [-p <data_path>]')
        sys.exit(2)
    random = False
    random_path = None
    array_id = None
    approach = None
    disease = None
    job_id = None
    given_path = None
    for i in opts:
        if i[0]=='-s':
            array_id=int(i[1])
        if i[0]=='-a':
            approach=i[1]
        if i[0]=='-d':
            disease=i[1]
        if i[0]=='-r':
            random=True
            random_path=i[1]
        if i[0]=='-j':
            job_id=i[1]
        if i[0]=='-p':
            given_path = i[1]
            if not os.path.exists(given_path):
                logger.error(f"Error: Path {given_path} doesn't exist.")
                sys.exit(2)
    if ((array_id == None)|(approach == None)|(disease==None)):
        print('Error: Missing argument. Usage: python linear_regression.py -s <array_id> -a <approach> -d <disease>')
        sys.exit(2)
    else:
        path = os.path.dirname(os.path.realpath(__file__))
        is_sge = (job_id!=None)
        log_path = path/Path('res_linear') if not random else path/Path('res_random')
        log_file = f'{log_path}/lin_reg_sge.sh.o{job_id}.{array_id+1}' if is_sge else f'{log_path}/{array_id}.out'
        logging.basicConfig(filename=log_file, filemode='a', force=True, format='%(asctime)s %(levelname)s - %(message)s')
        if given_path != None: path = given_path
        data_path = Path(path)/disease/approach
        if random:
            logger.info('Randomized')
            random_path = Path(random_path)
            if not os.path.exists(random_path):
                logger.error(f"Error: Path {random_path} doesn't exist.")
                sys.exit(2)
        #read in filtered transcript and miRNA expression from file
        transcript_counts = pd.read_pickle(data_path/'filtered_pivoted_transcript_counts.pickle').rename(columns={'sample':'transcript_id'})
        mirna_counts = pd.read_pickle(data_path/'filtered_miRNA_counts.pickle').rename(columns={'sample':'miRNA_id'})
        file = data_path/'reg_split_pairs_10000'/f'gene_mirna_pairs_{array_id}.pickle' if not random else random_path/'pairs'/f'gene_mirna_pairs_{array_id}.pickle'
        if file.is_file():
            gene_mirna_pairs = pd.read_pickle(file)
            logger.info(f'Approach {approach}: Run (gene,miRNA)-level linear regression on miRNA and transcript expression data.')
            run_linear_regression(data_path,mirna_counts,transcript_counts,gene_mirna_pairs,array_id,random,random_path) 
            logger.info('Finished.')
        else:
            logger.error(f"Error: file gene_mirna_pairs_{array_id}.pickle doesn't exist in folder split_filtered_pairs.")
            sys.exit(2)

if __name__=='__main__':
    main(sys.argv[1:])
