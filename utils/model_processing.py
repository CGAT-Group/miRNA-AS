import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import os
import scipy
from statsmodels.stats.multitest import fdrcorrection

def filter_models_by_rmse(data_path, type_, threshold=0.7, do_plot=False, prefix='',huge=False):
    file0 = data_path/type_/f'{prefix}filtered_models.pickle'
    if not file0.is_file():
        with open(data_path/type_/f'{prefix}models.pickle', 'rb') as f:
            models = pickle.load(f)
        logger.info(f'filter out {type_} models if RMSE >= {threshold}')
        logger.info(f'  Number models before: {len(models)}')
        before = [rmse_score for (rmse_score, regr, t_pvalues, f_pvalue) in models.values()]
        if huge: 
            with open(data_path/type_/f'{prefix}before.pickle', 'wb') as handle:
                pickle.dump(before, handle, protocol=pickle.HIGHEST_PROTOCOL)
        filtered_models = {mirna:values for mirna, values in models.items() if values[0] < threshold} 
        with open(file0, 'wb') as handle:
            pickle.dump(filtered_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'  Number models after: {len(filtered_models)}')
        if do_plot:
            #plot histogram of rooted mean squared error score
            fig1, ax1 = plt.subplots(figsize=(6,4))
            before = [b for b in before if b < 20]
            bins_before = np.arange(min(before),max(before),(max(before)-min(before))/100)
            plt.hist(before, bins=bins_before, alpha=0.4, color='tab:blue',label='before')
            plt.hist([rmse_score for (rmse_score, regr, t_pvalues, f_pvalue) in filtered_models.values()], bins=bins_before, alpha=0.5, color='tab:green',label='after')
            plt.ylabel(f"Number (miRNA, gene) {type_}")
            plt.xlabel("Rooted mean squared error")
            plt.axvline(x = threshold, color = 'black', linestyle="--",label='chosen threshold') 
            plt.legend()
            plt.savefig(data_path/'plots'/f'{type_}_rmse_filter.png',bbox_inches = "tight",dpi=200)
            plt.close()

    else:
        logger.info(f'Read in filtered {type_} models from file.')
        with (open(file0, "rb")) as fp:
            filtered_models = pickle.load(fp)
    return filtered_models

#what if different models kicked out -> get common pairs to do statistic on model pairs
#returns format {(mirna,gene):(nb_t_model, all_t_model)}
def get_common_models(data_path, nb_t_models, all_t_models, prefix=''):
    file = data_path/f'{prefix}filtered_common_models.pickle'
    if not file.is_file():
        common_pairs = {key:(data[1],all_t_models[key][1]) for key,data in nb_t_models.items() if key in all_t_models.keys()}
        with open(file, 'wb') as handle:
            pickle.dump(common_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'Keep {len(common_pairs)} common pairs.')
    else:
        logger.info('Read in common pairs from file.')
        with open(file, 'rb') as handle:
            common_pairs = pickle.load(handle)
    return common_pairs


#calculate log-likelihood per each model
def calc_ll(data_path, common_pairs, do_plot=False,suffix=''):
    file = data_path/f'log_likelihood{suffix}.pickle'
    if not os.path.exists(file):
        logger.info('Calculate log-likelihood for each (full_model,reduced_model)-pair.')
        log_lls = []
        for key, (nbt_model, all_t_model) in common_pairs.items():
            #calculate log-likelihood of models
            full_ll = all_t_model.llf
            reduced_ll = nbt_model.llf

            #calculate likelihood ratio Chi-Squared test statistic
            LR_statistic = -2*(reduced_ll-full_ll)

            #calculate p-value of test statistic using diff nr_t as degrees of freedom
            #DF = full_model(#patients + #transcripts) - reduced_model(#patients + #transcripts)
            #DF = # all transcripts - # nonbinding transcripts = # binding transcripts
            df_full = all_t_model.summary().as_text()
            df_full = df_full[df_full.find('Df Model:')+9:] #Df of model = # variables
            df_full = int(df_full[:df_full.find('\n')])

            df_reduced = nbt_model.summary().as_text()
            df_reduced = df_reduced[df_reduced.find('Df Model:')+9:] #Df of model = # variables
            df_reduced = int(df_reduced[:df_reduced.find('\n')])

            dof = df_full - df_reduced

            p_val = scipy.stats.chi2.sf(LR_statistic, dof)
            log_lls.append([key[0], key[1], reduced_ll, full_ll, LR_statistic, p_val, df_full, df_reduced, dof])

        log_lls = pd.DataFrame(log_lls,columns=['miRNA','gene','reduced_ll','full_ll','LR_statistic','p_val','dof_full','dof_reduced','dof'])
        #pvalue correction for false discovery rate (Benjamini Hochberg multiple testing correction)
        #log likelihood for m_0 (H_0) must be <= log likelihood of m_1 (H_1)
        #in some cases reduced model has same/higher log-likelihood -> LR_statistic <= 0 -> p-value nan
        logger.info(f'For {len(log_lls[log_lls.p_val.isna()])} pairs the reduced model has the same log-likelihood as the full model.')
        #we skip those cases for calculation
        log_lls['rejected'], log_lls['corr_p_val'] = False, np.nan
        not_na = log_lls[~log_lls.p_val.isna()].copy()
        not_na['rejected'], not_na['corr_p_val'] = fdrcorrection(not_na.p_val.values,alpha=0.05)
        log_lls = log_lls[log_lls.p_val.isna()]
        log_lls = pd.concat([log_lls,not_na])
        log_lls.sort_index()
        log_lls.to_pickle(file)
        if do_plot:
            #ll results plot
            columns = ['reduced_ll','full_ll','LR_statistic','p_val','dof_full','dof_reduced','dof','corr_p_val']
            labels = ['Log-likelihood of reduced model','Log-likelihood of full model','Likelihood ratio Chi-Squared test statistic','P-value for test statistic','Degrees of freedom full model','Degrees of freedom reduced model','Difference in degrees of freedom between reduced and full model','Benjamini Hochberg corrected p-value']
            for idx,col in enumerate(columns):
                log_lls[col].hist()
                plt.ylabel("Amount of (miRNA, gene) models")
                plt.xlabel(labels[idx])
                if (col == 'p_val') | (col == 'corr_p_val'):
                    plt.axvline(x = 0.05, color = 'black', linestyle="--",label='threshold') 
                    plt.legend()
                plt.savefig(data_path/'plots'/f'{col}_hist{suffix}.png',bbox_inches = "tight",dpi=200)
                plt.close()
    else:
        logger.info('Read in log-likelihood for each (full_model,reduced_model)-pair.')
        log_lls = pd.read_pickle(file)
    return log_lls