import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from os import listdir
import numpy as np
import math
import scipy
import sys
import getopt
import subprocess
import os.path
import time
from gtfparse import read_gtf
import gc
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import urllib.request
import gzip
from statsmodels.stats.multitest import fdrcorrection
import shutil
from utils import file_operations
from utils import model_processing

#read in TarPmiR binding site predictions with cutoff
def get_tarpmir_bs(input_path, cutoff=0.5):
    file = input_path/f'bs_transcript_level_{cutoff}.pickle'
    if not file.is_file():
        bs = pd.read_parquet(input_path/'bs_tarp_all.parquet')
        bs.binding_probability.hist()
        if cutoff != 0.5:
            bs = bs[bs.binding_probability > cutoff]                     
        logger.info(f'{len(bs)} TarPmiR binding sites have probability > {cutoff}.')
        bs_transcript_level = bs.rename(columns={'mRNA':'ensembl_transcript_id'})[['miRNA','ensembl_transcript_id']].drop_duplicates()
        t2g = get_map_g2t(input_path,g2t=False)
        bs_transcript_level = bs_transcript_level.merge(t2g, left_on='ensembl_transcript_id', right_on='transcript_id',how='left').rename(columns={'gene_id':'ensembl_gene_id'})
        with open(file, "wb") as fp:
            pickle.dump(bs_transcript_level, fp)
    else:
        with (open(file, "rb")) as fp:
            bs_transcript_level = pickle.load(fp)
    return bs_transcript_level

#map TCGA samples to sample type and cancer type
def get_mapping_sample_to_cancer(input_path):
    if not Path(input_path/'unaligned.tsv').is_file():
        url = 'https://github.com/jvivian/ipython_notebooks/raw/master/tcga_rnaseq_analysis/unaligned.tsv'
        urllib.request.urlretrieve(url,input_path/'unaligned.tsv')
    sample_map = pd.read_csv(input_path/'unaligned.tsv', sep='\t')
    return sample_map

#filter out expression samples from other diseases than the one of interest and provide a list of the remaining samples
def filter_expression_by_disease(input_path, transcript_counts, disease, sample_map, type_):
    logger.info(f'filter out samples from other diseases than {disease}')
    logger.info(f'  Number {type_} samples before: {len(transcript_counts.columns)}')
    sample_map['sample'] = sample_map['barcode'].str[0:15]
    if disease in ['ILC','IDC']:
        s1 = pd.read_csv(input_path/'s1.csv')[['CLID','2016 Histology Annotations']].rename(columns={'CLID':'sample_id','2016 Histology Annotations':'cancer_type'})
        s1['sample'] = s1.sample_id.apply(lambda x: x[:-1])
        s1 = s1[s1.cancer_type == ('Invasive lobular carcinoma' if (disease == 'ILC') else 'Invasive ductal carcinoma')] #IDC 652, ILC 184
        sample_map = sample_map[(sample_map.disease == 'BRCA') & (sample_map['sample'].apply(lambda x: x in list(set(s1['sample']))))] 
    else:
        sample_map = sample_map[sample_map.disease == disease]
    disease_samples = list(set(sample_map['sample']))
    disease_samples = [sample for sample in disease_samples if sample in transcript_counts.columns]
    disease_samples.append('sample')
    transcript_counts = transcript_counts[disease_samples]
    logger.info(f'  Number {type_} samples after: {len(transcript_counts.columns)}')
    return transcript_counts, sample_map

#https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
#filter out samples with types Additional - New Primary (5), Metastatic (6), Additional Metastatic (7), EBV Immortalized Normal (13), sample type 15 (15), sample type 16 (16), Cell Lines (50), Primary Xenograft Tissue (60), Cell Line Derived Xenograft Tissue (61), sample type 99 (99) and also change list of samples accordingly
def filter_expression_by_sampletype(nested_path, transcript_counts, sample_map, type_):
    logger.info(f'filter out {type_} samples with a wrong sample_type')
    logger.info(f'  Number samples before: {len(transcript_counts.columns)}')
    sample_map['keep_sample'] = sample_map.sample_type_name.apply(lambda x: x not in [5, 6, 7, 13, 15, 16, 50, 60, 61, 99])
    sample_map = sample_map[sample_map['keep_sample']].drop('keep_sample',axis=1)
    sample_map['sample'] = sample_map['barcode'].str[0:15]
    sampletype_samples = list(set(sample_map['sample']))
    sampletype_samples = [smpl for smpl in sampletype_samples if smpl in transcript_counts.columns]
    sample_map['in'] = sample_map.apply(lambda x: x['sample'] in sampletype_samples,axis=1)
    sample_map = sample_map[sample_map['in']]
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sample_map.sample_type.value_counts().plot(kind='bar',ax=ax1)
    plt.xlabel("Sample types")
    plt.ylabel(f"Amount of {type_} samples")
    plt.savefig(nested_path/'plots'/f'sample_types_{type_}.png', bbox_inches = "tight",dpi=200)
    plt.close()
    sampletype_samples.append('sample')
    transcript_counts = transcript_counts[sampletype_samples]
    logger.info(f'  Number {type_} samples after: {len(transcript_counts.columns)}')
    return transcript_counts, sample_map

#download TCGA hg38 transcript expression data from XENA 
#filter out samples from other diseases or with wrong sample types and remove version from Ensembl transcript id
#https://xenabrowser.net/datapages/?dataset=tcga_Kallisto_tpm&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=http%3A%2F%2F127.0.0.1%3A7222&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
def get_transcript_expression(nested_path, input_path, disease):
    file = nested_path/('transcript_counts.parquet')
    if not file.is_file():
        if not Path(input_path/'tcga_Kallisto_tpm').is_file():
            logger.info(f'Download transcript expression data from TCGA')
            url = 'https://toil.xenahubs.net/download/tcga_Kallisto_tpm.gz'
            urllib.request.urlretrieve(url,input_path/'tcga_Kallisto_tpm.gz')
            with gzip.open(input_path/'tcga_Kallisto_tpm.gz', 'rb') as f_in:
                with open(input_path/'tcga_Kallisto_tpm', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(input_path/'tcga_Kallisto_tpm.gz')
        transcript_counts = pd.read_csv(input_path/'tcga_Kallisto_tpm', sep='\t')
        sample_map = get_mapping_sample_to_cancer(input_path)
        transcript_counts, sample_map = filter_expression_by_disease(input_path,transcript_counts, disease, sample_map,'transcript')
        transcript_counts, sample_map = filter_expression_by_sampletype(nested_path, transcript_counts, sample_map,'transcript')
        transcript_counts.to_parquet(file)
    else:
        logger.info('read transcript expression from file')
        transcript_counts = pd.read_parquet(file)
    transcript_counts['sample'] = transcript_counts['sample'].str.split('.',expand=True)[0]
    return transcript_counts 

#download TCGA hg38 already preprocessed HiSeq + GA data together miRNA expression data from XENA, 743 miRNAs × 10.824 patients
#filter out samples from other diseases or with wrong sample types
#https://xenabrowser.net/datapages/?dataset=pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena&host=https%3A%2F%2Fpancanatlas.xenahubs.net&removeHub=http%3A%2F%2F127.0.0.1%3A7222&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
def download_mirna_expression(nested_path, input_path, disease):
    file = nested_path/('test_mirna_counts.pickle')
    if not file.is_file():
        if not Path(input_path/'tcga_miRNA').is_file():
            logger.info('Download miRNA expression data from TCGA')
            url = 'https://pancanatlas.xenahubs.net/download/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz'
            urllib.request.urlretrieve(url,input_path/'tcga_miRNA.gz')         
            with gzip.open(input_path/'tcga_miRNA.gz', 'rb') as f_in:
                with open(input_path/'tcga_miRNA', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(input_path/'tcga_miRNA.gz')
        mirna_counts = pd.read_csv(input_path/'tcga_miRNA', sep='\t')
        sample_map = get_mapping_sample_to_cancer(input_path)
        mirna_counts, sample_map = filter_expression_by_disease(input_path,mirna_counts, disease, sample_map,'miRNA')
        mirna_counts, sample_map = filter_expression_by_sampletype(nested_path, mirna_counts, sample_map,'miRNA')
        mirna_counts.to_parquet(file)
    else:
        logger.info('read miRNA expression from file')
        mirna_counts = pd.read_parquet(file)
    return mirna_counts 

#variance filter: drop miRNA expression that shows no variance (not on gene level but on all caseids)
def filter_no_variance(mirna_counts,nested_path,common_caseids,threshold=0.2,do_plot=True):
    logger.info(f'filter out mirnas with variance <= {threshold}')
    logger.info(f'  Number mirnas before: {len(mirna_counts)}')
    mirna_counts = mirna_counts[common_caseids]
    mirna_counts['var'] = np.var(mirna_counts.loc[:, mirna_counts.columns != 'sample'],axis=1)
    before = mirna_counts['var'].copy()
    if do_plot:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        mirna_counts['var'].plot(kind='hist', bins=36,ax=ax1)
        plt.ylabel("Amount of miRNAs") 
        plt.xlabel("Expression variance between samples")
        plt.savefig(nested_path/'plots'/'mirna_var_filter.png', bbox_inches = "tight",dpi=200)
        plt.close()
        fig1, axs = plt.subplots(2,1,figsize=(5,6))
        fig1.tight_layout(h_pad=3)
        mirna_counts['var'].plot(kind='hist', bins=72, cumulative=True,ax=axs[0])
        axs[0].set_ylabel("Amount of miRNAs < threshold")
        axs[0].set_xlabel("Expression variance threshold between samples")
        mirna_counts['var'].plot(kind='hist', bins=288, cumulative=True,ax=axs[1],label='')
        axs[1].set_xlim([0,1])
        axs[1].set_ylabel("Amount of miRNAs < threshold")
        axs[1].set_xlabel("Expression variance threshold between samples")
        plt.axvline(x = threshold, color = 'black', linestyle="--",label='chosen threshold') 
        plt.legend()
        plt.savefig(nested_path/'plots'/'mirna_var_filter_thresh.png', bbox_inches = "tight",dpi=400)
        plt.close()
    mirna_counts = mirna_counts[mirna_counts['var'] > threshold]
    if do_plot:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        before.plot(kind='hist', bins=np.arange(0,int(max(before)),0.2),ax=ax1, alpha=0.4, color='tab:blue',label='before')
        mirna_counts['var'].plot(kind='hist', bins=np.arange(0,int(max(before)),0.2),ax=ax1, alpha=0.5, color='tab:green',label='after')
        plt.ylabel("Amount of miRNAs")
        plt.xlabel("Expression variance between samples")
        plt.legend()
        plt.savefig(nested_path/'plots'/'mirna_var_filter2.png', bbox_inches = "tight",dpi=200)
        plt.close()
    mirna_counts = mirna_counts.drop(columns='var')
    logger.info(f'  Number mirnas after: {len(mirna_counts)}')
    return mirna_counts

#get miRNA expression after removing miRNAs with little variation for disease
def get_mirna_expression(nested_path, input_path, disease):
    file = Path(nested_path/'miRNA_counts.pickle')
    if file.is_file():
        logger.info('read miRNA expression from files')
        mirna_counts = pd.read_pickle(file)
    else:
        logger.info(f'calculate miRNA expression for {disease}')
        mirna_counts = download_mirna_expression(nested_path, input_path, disease)
        transcript_counts = get_transcript_expression(nested_path, input_path, disease)
        common_caseids = list(set(transcript_counts.columns) & set(mirna_counts.columns))
        mirna_counts = filter_no_variance(mirna_counts,nested_path,common_caseids)
        mirna_counts.to_pickle(file)
    mirna_counts = mirna_counts.melt(id_vars=['sample'], var_name="case_id", value_name="mirna_count").dropna()
    return mirna_counts

#get TCGA hg38 gene expression data from XENA (RSEM) #TODO problem? for transcript expression we used Kallisto but here not available
#https://xenabrowser.net/datapages/?dataset=tcga_RSEM_gene_tpm&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=http%3A%2F%2F127.0.0.1%3A7222
def get_gene_expression(nested_path, input_path, disease):
    file = nested_path/('gene_counts.parquet')
    if not file.is_file():
        if not Path(input_path/'tcga_RSEM_gene_tpm').is_file():
            logger.info(f'Download gene expression data from TCGA')
            url = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_gene_tpm.gz'
            urllib.request.urlretrieve(url,input_path/'tcga_RSEM_gene_tpm.gz')
            with gzip.open(input_path/'tcga_RSEM_gene_tpm.gz', 'rb') as f_in:
                with open(input_path/'tcga_RSEM_gene_tpm', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(input_path/'tcga_RSEM_gene_tpm.gz')
        gene_counts = pd.read_csv(input_path/'tcga_RSEM_gene_tpm', sep='\t')
        sample_map = get_mapping_sample_to_cancer(input_path)
        gene_counts, sample_map = filter_expression_by_disease(input_path,gene_counts, disease, sample_map,'gene')
        gene_counts, sample_map = filter_expression_by_sampletype(nested_path, gene_counts, sample_map,'gene')
        gene_counts['sample'] = gene_counts['sample'].str.split('.', expand=True)[0]
        gene_counts = gene_counts.melt(id_vars=['sample'], var_name="case_id", value_name="gene_count")
        gene_counts = gene_counts.pivot_table(index=['case_id'], columns=['sample'], values = 'gene_count')
        gene_counts.to_parquet(file)
    else:
        logger.info('read gene expression from file')
        gene_counts = pd.read_parquet(file)
    return gene_counts 

#get mapping gene_id to transcript_ids or transcript_id to gene_id
def get_map_g2t(input_path,g2t=True):
    file = input_path/'transcript_map.parquet'
    if not file.is_file():
        if not Path(input_path/"gencode.v23.chr_patch_hapl_scaff.annotation.gtf").is_file():
            url = 'ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_23/gencode.v23.chr_patch_hapl_scaff.annotation.gtf.gz'
            urllib.request.urlretrieve(url,input_path/'gencode.v23.chr_patch_hapl_scaff.annotation.gtf.gz')
            with gzip.open(input_path/'gencode.v23.chr_patch_hapl_scaff.annotation.gtf.gz', 'rb') as f_in:
                with open(input_path/'gencode.v23.chr_patch_hapl_scaff.annotation.gtf', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(input_path/'gencode.v23.chr_patch_hapl_scaff.annotation.gtf.gz')
        transcript_map = read_gtf(input_path/"gencode.v23.chr_patch_hapl_scaff.annotation.gtf")
        transcript_map = transcript_map[transcript_map.feature == 'exon']
        transcript_map.to_parquet(file)
    else: 
        transcript_map = pd.read_parquet(file)
    transcript_map = transcript_map[['transcript_id','gene_id']].drop_duplicates()
    transcript_map['transcript_id'] = transcript_map['transcript_id'].str.split('.',expand=True)[0]
    transcript_map['gene_id'] = transcript_map['gene_id'].str.split('.',expand=True)[0]
    if g2t:
        transcript_map = transcript_map.groupby('gene_id').aggregate({'transcript_id': lambda x : list(x)}).rename(columns={'transcript_id':'all_transcripts'})
    return transcript_map

#filter miRNA, transcript and gene expression for common samples 
def get_common_samples(mirna_counts, transcript_counts, gene_counts):
    com_samples = set(mirna_counts.case_id.unique()) & set(transcript_counts.case_id.unique())
    mirna_counts = mirna_counts[mirna_counts.case_id.isin(com_samples)]
    transcript_counts = transcript_counts[transcript_counts.case_id.isin(com_samples)]
    gene_counts = gene_counts[gene_counts.case_id.isin(com_samples)]
    logger.info(f'Keep {len(com_samples)} common samples for gene, transcript and miRNA expression.')
    return mirna_counts, transcript_counts, gene_counts

#return TarPmiR target sites on transcript level, miRNA, transcript, gene expression for common samples
def get_common_with_bs(nested_path, input_path, disease):
    file = nested_path/'filtered_gene_counts_old.pickle'
    if not file.is_file():
        #read in mirna, transcript, gene expression and TarPmiR bindingsites
        bs_transcript_level = get_tarpmir_bs(input_path, cutoff=0.8)
        mirna_counts = get_mirna_expression(nested_path, input_path, disease)
        transcript_counts = get_transcript_expression(nested_path, input_path, disease)
        gene_counts = get_gene_expression(nested_path, input_path, disease)

        nr_mirna_before = len(mirna_counts['sample'].unique())
        mirnas_com = bs_transcript_level['miRNA'].to_frame().drop_duplicates().merge(mirna_counts['sample'].drop_duplicates(), how='inner', left_on='miRNA',right_on='sample')
        mirna_counts = mirna_counts[mirna_counts['sample'].isin(mirnas_com.miRNA)]

        nr_gene_before = len(gene_counts.columns)
        genes_com = bs_transcript_level['ensembl_gene_id'].to_frame().drop_duplicates().merge(gene_counts.columns.to_frame().reset_index(drop=True), how='inner', left_on='ensembl_gene_id',right_on='sample')
        gene_counts = gene_counts[genes_com.ensembl_gene_id]

        nr_t_before = len(bs_transcript_level)
        bs_transcript_level = bs_transcript_level[bs_transcript_level['ensembl_gene_id'].isin(genes_com.ensembl_gene_id)]
        logger.info(f"We keep only {len(bs_transcript_level)}/{nr_t_before} transcripts because we don't have the according gene expressions.")

        #kick out transcript where there is no gene in miRNA-transript pairs because bs_transcript_level only contains binding transcripts for now
        nr_transcript_before = len(transcript_counts)
        t2g = get_map_g2t(input_path,g2t=False)
        t2g = t2g[t2g.gene_id.isin(genes_com.ensembl_gene_id)]
        transcript_counts = transcript_counts[transcript_counts['sample'].isin(t2g.transcript_id)]

        bs_transcript_level = bs_transcript_level[(bs_transcript_level['ensembl_transcript_id'].isin(transcript_counts['sample'].unique())) & (bs_transcript_level['miRNA'].isin(mirnas_com.miRNA))]

        logger.info(f'Keeping {len(mirna_counts["sample"].unique())}/{nr_mirna_before} miRNAs, {len(transcript_counts)}/{nr_transcript_before} transcripts and {len(gene_counts.columns)}/{nr_gene_before} genes.')
        #melt transcript_counts
        transcript_counts = transcript_counts.melt(id_vars=['sample'],value_vars=list(transcript_counts.columns)[:-1],var_name='case_id',value_name='transcript_count')
        gene_counts = gene_counts.reset_index().melt(id_vars=['case_id'],value_vars=list(gene_counts.columns),var_name='sample',value_name='gene_count')
        bs_transcript_level.to_pickle(nested_path/'filtered_bs_transcript_level.pickle')
        mirna_counts, transcript_counts, gene_counts = get_common_samples(mirna_counts, transcript_counts, gene_counts)
        gene_counts.to_pickle(nested_path/'filtered_gene_counts_old.pickle')
        transcript_counts.to_pickle(nested_path/'filtered_transcript_counts_old.pickle')
        mirna_counts.to_pickle(nested_path/'filtered_miRNA_counts_old.pickle')
    else:
        gene_counts = pd.read_pickle(nested_path/'filtered_gene_counts_old.pickle')
        transcript_counts = pd.read_pickle(nested_path/'filtered_transcript_counts_old.pickle')
        mirna_counts = pd.read_pickle(nested_path/'filtered_miRNA_counts_old.pickle')
        bs_transcript_level = pd.read_pickle(nested_path/'filtered_bs_transcript_level.pickle')
        logger.info(f'{len(mirna_counts["sample"].unique())} miRNAs, {len(transcript_counts["sample"].unique())} transcripts and {len(gene_counts["sample"].unique())} genes x {len(gene_counts["case_id"].unique())} samples left after getting common.')
    return bs_transcript_level, mirna_counts, transcript_counts, gene_counts

#calculate transcript binding types for miRNA-transcript pairs using TarPmiR binding sites (disease independent)
def calc_transcript_types(path, input_path, do_plots):
    file0=input_path/'transcript_binding_types.pickle'
    if not file0.is_file():
        logger.info('Calculate transcript types using TarPmiR binding sites.')
        file=input_path/'transcript_mirna_binding.pickle'
        if not file.is_file():
            logger.info('Calculate overlap coding region with TarPmiR binding sites.')
            # read in TarPmiR predicted miRNA binding sites
            mirna_bs = pd.read_parquet(input_path/'bs_tarp_all.parquet')
            mirna_bs = mirna_bs[['miRNA','mRNA','binding_probability','bs_start','bs_end']].rename(columns={'mRNA':'transcript_id','miRNA':'mirna_id'})

            #tarpmir 0 based, cds 1 based -> both 1 based
            mirna_bs['bs_start'] = mirna_bs['bs_start'] + 1
            mirna_bs['bs_end'] = mirna_bs['bs_end'] + 1

            #read in transcript features like CDS start and end (1.039.716 lines) downloaded from Ensembl website
            file = input_path/'transcript_CDS.pickle'
            if not file.is_file():
                logger.info('Recalculate transcript features.')
                transcripts = pd.read_csv(input_path/'tlevel_cDNA_coding.human.GRCh38.p13.csv')
                transcripts = transcripts.rename(columns={'Transcript stable ID':'transcript_id','cDNA coding start':'CDS_start','cDNA coding end':'CDS_end'})
                transcripts = transcripts.groupby(['transcript_id']).aggregate({'CDS_start':min,'CDS_end':max}).reset_index()
                transcripts.to_pickle(file)
            else:
                logger.info('Read in transcript features from file.')
                transcripts = pd.read_pickle(file)
            mirna_bs = transcripts.merge(mirna_bs)

            #take only noncoding transcripts
            mirna_bs['coding'] = ~mirna_bs.CDS_start.isna()
            noncod_bs = mirna_bs[~mirna_bs['coding']].copy()
            noncod_bs['CDS_bs_50'] = False
            noncod_bs['NCDS_bs_50'] = True
            noncod_bs['CDS_bs_80'] = False
            noncod_bs['NCDS_bs_80'] = mirna_bs.binding_probability > 0.8
            noncod_bs = noncod_bs.groupby(['mirna_id','transcript_id']).aggregate({'CDS_bs_50':sum,'NCDS_bs_50':sum,'CDS_bs_80':sum,'NCDS_bs_80':sum,'coding':'first'})

            #only take coding transcripts & categorize bs
            mirna_bs = mirna_bs[mirna_bs['coding']]
            bs_start_in_CDS = (mirna_bs.bs_start >= mirna_bs.CDS_start) & (mirna_bs.bs_start < mirna_bs.CDS_end)
            bs_end_in_CDS = (mirna_bs.bs_end > mirna_bs.CDS_start) & (mirna_bs.bs_end <= mirna_bs.CDS_end)
            bs_before_CDS = mirna_bs.bs_start < mirna_bs.CDS_start
            bs_after_CDS = mirna_bs.bs_end > mirna_bs.CDS_end
            mirna_bs['CDS_bs_50'] = bs_start_in_CDS | bs_end_in_CDS
            mirna_bs['NCDS_bs_50'] = bs_before_CDS | bs_after_CDS
            mirna_bs['CDS_bs_80'] = (mirna_bs.binding_probability > 0.8) & mirna_bs.CDS_bs_50
            mirna_bs['NCDS_bs_80'] = (mirna_bs.binding_probability > 0.8) & mirna_bs.NCDS_bs_50

            #add coding & noncoding back together
            cod_bs = mirna_bs.groupby(['mirna_id','transcript_id']).aggregate({'CDS_bs_50':sum,'NCDS_bs_50':sum,'CDS_bs_80':sum,'NCDS_bs_80':sum,'coding':'first'})
            mirna_bs = cod_bs.append(noncod_bs)
            mirna_bs.to_pickle(file)

            if do_plots:
                mirna_bs.CDS_bs_50.hist(bins=30)
                plt.xlabel(f"Number bindingsites in coding region with p > 0.5")
                plt.ylabel(f"Number (transcript, miRNA)-pairs")
                plt.savefig(path/'plots'/f'CDS_bs_50_hist.png',bbox_inches = "tight",dpi=200)
                plt.close()
                mirna_bs.NCDS_bs_50.hist(bins=30)
                plt.xlabel(f"Number bindingsites in noncoding region with p > 0.5")
                plt.ylabel(f"Number (transcript, miRNA)-pairs")
                plt.savefig(path/'plots'/f'NCDS_bs_50_hist.png',bbox_inches = "tight",dpi=200)
                plt.close()
                mirna_bs.CDS_bs_80.hist(bins=30)
                plt.xlabel(f"Number bindingsites in coding region with p > 0.8")
                plt.ylabel(f"Number (transcript, miRNA)-pairs")
                plt.savefig(path/'plots'/f'CDS_bs_80_hist.png',bbox_inches = "tight",dpi=200)
                plt.close()
                mirna_bs.NCDS_bs_80.hist(bins=30)
                plt.xlabel(f"Number bindingsites in noncoding region with p > 0.8")
                plt.ylabel(f"Number (transcript, miRNA)-pairs")
                plt.savefig(path/'plots'/f'NCDS_bs_80_hist.png',bbox_inches = "tight",dpi=200)
                plt.close()
        else:
            logger.info('Read in overlap coding region with TarPmiR binding sites from file.')
            mirna_bs = pd.read_pickle(file)

        def get_binding_type(x):
            if x.CDS_bs_50 == 0:
                if (x.NCDS_bs_50 == 0):
                    return 1
                elif (x.NCDS_bs_80 > 0):
                    return 0
                else:
                    return None
            elif x.CDS_bs_80 > 0:
                if x.NCDS_bs_50 == 0:
                    return 2
                elif x.NCDS_bs_80 > 0:
                    return 3
                else: 
                    return None
            else:
                return None
        mirna_bs['binding_type'] = mirna_bs.apply(lambda x: get_binding_type(x),axis=1)
        mirna_bs.to_pickle(file0)
        if do_plots:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for x in range(4):
                bar = ax.bar(x,len(mirna_bs[mirna_bs.binding_type == x]),label=f'Type {x}')
                # Add counts above the bar graphs
                for rect in bar:
                    height = rect.get_height()
                    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
            plt.ylabel("Number transcripts")
            plt.legend()
            plt.savefig(path/'plots'/f'binding_types_hist.png',bbox_inches = "tight",dpi=200)
            plt.close()

    else:
        logger.info('Read in transcript binding types from file.')
        mirna_bs = pd.read_pickle(file0)
    return mirna_bs

#keep only miRNA-gene pairs that have more than 1 transcript, where at least 1 transcript has a bindingsite with p>0.8 & another transcript has no bindingsite with p>0.5
def alternative_splicing_filter(path, input_path, bs_t_level, gene_counts, transcript_counts, mirna_counts, approach):
    transcript_mirna_bs = calc_transcript_types(path, input_path, do_plots=True)
    bs_t_level = bs_t_level.merge(transcript_mirna_bs,left_on=['miRNA','ensembl_transcript_id'],right_on=['mirna_id','transcript_id'], how='left')
    #it's important to NOT remove binding_type nan

    #keep only type 1 & 2
    if approach == 'A':
        t_with_maybe_bs = bs_t_level[bs_t_level.binding_type != 1].groupby(['ensembl_gene_id','miRNA']).aggregate({'ensembl_transcript_id':list}).rename(columns={'ensembl_transcript_id':'other_bs'})
        gene_mirna_pairs = bs_t_level[bs_t_level.binding_type == 2].groupby(['ensembl_gene_id','miRNA']).aggregate({'ensembl_transcript_id':list}).rename(columns={'ensembl_transcript_id':'bs'})
        # binding_type 1 should be empty as TarPmiR results only contain bindingsites -> no transcripts without bs

    #keep only type 0 & 3
    elif approach == 'B':
        bs = bs_t_level[bs_t_level.binding_type == 3].groupby(['ensembl_gene_id','miRNA']).aggregate({'ensembl_transcript_id':list}).rename(columns={'ensembl_transcript_id':'bs'})
        nbs = bs_t_level[bs_t_level.binding_type == 0].groupby(['ensembl_gene_id','miRNA']).aggregate({'ensembl_transcript_id':list}).rename(columns={'ensembl_transcript_id':'nbs'})
        gene_mirna_pairs = bs.merge(nbs,  left_index=True, right_index=True, how='inner')
    
    #keep all types
    elif approach == 'C':
        t_with_maybe_bs = bs_t_level[bs_t_level.binding_type != 1].groupby(['ensembl_gene_id','miRNA']).aggregate({'ensembl_transcript_id':list}).rename(columns={'ensembl_transcript_id':'other_bs'})
        # binding_type 1 should be empty as TarPmiR results only contain bindingsites -> no transcripts without bs
        gene_mirna_pairs = bs_t_level[(bs_t_level.binding_type == 0) | (bs_t_level.binding_type == 2) | (bs_t_level.binding_type == 3)].groupby(['ensembl_gene_id','miRNA']).aggregate({'ensembl_transcript_id':list}).rename(columns={'ensembl_transcript_id':'bs'})
        
    gene_mirna_pairs = gene_mirna_pairs.reset_index()

    prev_nr_pairs = len(gene_mirna_pairs)
    prev_nr_genes = len(gene_mirna_pairs.ensembl_gene_id.unique())
    prev_nr_transcripts = len(transcript_counts['sample'].unique())
    prev_nr_mirnas = len(gene_mirna_pairs.miRNA.unique())

    #only keep t in gene-mirna-pair where we have transcripts_counts for
    t2g = get_map_g2t(input_path,g2t=False)
    g2t = t2g.merge(transcript_counts['sample'],left_on='transcript_id', right_on='sample', how='right').groupby('gene_id').aggregate({'sample':list}).rename(columns={'sample':'transcripts_with_data'})
    gene_mirna_pairs = gene_mirna_pairs.merge(g2t,left_on='ensembl_gene_id',right_on='gene_id',how='left')

    #keep only type 1 & 2
    if approach == 'A':
        #to get all_transcripts
        g2t = get_map_g2t(input_path)
        gene_mirna_pairs = gene_mirna_pairs.merge(g2t,left_on='ensembl_gene_id',right_on='gene_id',how='left')
        gene_mirna_pairs = gene_mirna_pairs.merge(t_with_maybe_bs, on=['ensembl_gene_id','miRNA'],how='left')
        gene_mirna_pairs['nbs'] = gene_mirna_pairs.apply(lambda x: [t for t in list(set(x.all_transcripts)-set(x.other_bs)) if t in x.transcripts_with_data],axis=1)

    #keep only type 0 & 3
    elif approach == 'B':
        gene_mirna_pairs['nbs'] = gene_mirna_pairs.apply(lambda x: [t for t in x.nbs if t in x.transcripts_with_data],axis=1)
    
    #keep only type 1 & 0,2,3
    elif approach == 'C':
        #to get all_transcripts
        g2t = get_map_g2t(input_path)
        gene_mirna_pairs = gene_mirna_pairs.merge(g2t,left_on='ensembl_gene_id',right_on='gene_id',how='left')
        gene_mirna_pairs = gene_mirna_pairs.merge(t_with_maybe_bs, on=['ensembl_gene_id','miRNA'],how='left')
        gene_mirna_pairs['nbs'] = gene_mirna_pairs.apply(lambda x: [t for t in list(set(x.all_transcripts)-set(x.other_bs)) if t in x.transcripts_with_data],axis=1)

    gene_mirna_pairs['bs'] = gene_mirna_pairs.apply(lambda x: [t for t in x.bs if t in x.transcripts_with_data],axis=1)

    gene_mirna_pairs = gene_mirna_pairs[gene_mirna_pairs.apply(lambda x: (len(x.bs) > 0) & (len(x.nbs) > 0),axis=1)]

    nbt = gene_mirna_pairs.nbs.explode('nbs').rename({'nbs':'transcript_id'})
    bt = gene_mirna_pairs.bs.explode('bs').rename({'bs':'transcript_id'})
    all_t = pd.concat([nbt,bt]).drop_duplicates()
    gene_counts = gene_counts[gene_counts['sample'].isin(gene_mirna_pairs.ensembl_gene_id.unique())]
    transcript_counts = transcript_counts[transcript_counts['sample'].isin(all_t)]
    mirna_counts = mirna_counts[mirna_counts['sample'].isin(gene_mirna_pairs.miRNA.unique())]
    nr_genes = len(gene_mirna_pairs.ensembl_gene_id.unique())
    nr_transcripts = len(transcript_counts["sample"].unique())
    nr_mirnas = len(gene_mirna_pairs.miRNA.unique())
    logger.info(f'After alternative splicing filter: {len(gene_mirna_pairs)}/{prev_nr_pairs} ({(len(gene_mirna_pairs)/prev_nr_pairs)*100}%) (miRNA,gene) pairs left, {nr_genes}/{prev_nr_genes} ({(nr_genes/prev_nr_genes)*100}%) genes, {nr_transcripts}/{prev_nr_transcripts} ({(nr_transcripts/prev_nr_transcripts)*100}%) transcripts and {nr_mirnas}/{prev_nr_mirnas} ({(nr_mirnas/prev_nr_mirnas)*100}%) miRNAs left.')
    return gene_mirna_pairs, gene_counts, transcript_counts, mirna_counts

# mark low transcript expression < thres for >= 25% of samples
def low_transcript_expression_filter(data_path, transcript_counts, thres_t=math.log(0.001,2),do_plots=False):
    #unit transcript expression log2(tpm+0.001)
    transcript_counts['above_tres'] = transcript_counts['transcript_count'].apply(lambda x: x > thres_t)
    transcript_level = transcript_counts.groupby('sample').aggregate({'above_tres':sum}).reset_index()
    nr_patients = len(transcript_counts['case_id'].unique())
    if do_plots:
        transcript_level.above_tres.hist()
        plt.title(f'Distribution of number good sample expressions')
        plt.xlabel(f"Number samples with expression > thres (of total {transcript_level.above_tres.max()} samples)")
        plt.ylabel(f"Number transcripts (of total {len(transcript_level)})")
        plt.axvline(x=nr_patients*0.25, linestyle='--')
        plt.savefig(data_path/'plots'/f'low_transcript_filter_thres.png',bbox_inches = "tight",dpi=200)
        plt.close()

        transcript_counts.transcript_count.hist()
        plt.title('Distribution of transcript expression')
        plt.xlabel("Transcript expression")
        plt.ylabel("Number transcripts")
        plt.axvline(x=thres_t, linestyle='--')
        plt.savefig(data_path/'plots'/f'low_transcript_filter.png',bbox_inches = "tight",dpi=200)
        plt.close()

    nr_transcript_before = len(transcript_level)
    transcript_level = transcript_level[transcript_level.above_tres >= nr_patients*0.75]
    transcript_counts['enough_good_samples'] = transcript_counts['sample'].isin(transcript_level['sample'].values)
    logger.info(f'After checking expression > {thres_t} for >= 25% of samples: {len(transcript_level)}/{nr_transcript_before} transcripts marked as enough expression.')
    return transcript_counts

# filter out low gene expression < thres for >= 25% of samples
# keep only genes where we have high enough expression data for at least 1 nonbinding transcript & 1 binding transcript per gene
def low_gene_expression_filter(input_path, data_path, gene_mirna_pairs, transcript_counts, gene_counts, thres_t=math.log(0.001,2),do_plots=False):
    #unit transcript expression log2(tpm+0.001)
    nr_patients = len(gene_counts.case_id.unique())
    gene_counts['above_tres'] = gene_counts['gene_count'].apply(lambda x: x > thres_t)
    gene_level = gene_counts.groupby('sample').aggregate({'above_tres':sum}).reset_index()
    if do_plots:
        gene_level.above_tres.hist()
        plt.title(f'Distribution of number good sample expressions')
        plt.xlabel(f"Number samples with expression > thres (of total {gene_level.above_tres.max()} samples)")
        plt.ylabel(f"Number genes (of total {len(gene_level)})")
        plt.axvline(x=nr_patients*0.25, linestyle='--')
        plt.savefig(data_path/'plots'/f'low_gene_filter_thres.png',bbox_inches = "tight",dpi=200)
        plt.close()
        gene_counts.gene_count.hist()
        plt.title('Distribution of gene expression')
        plt.xlabel("Gene expression")
        plt.ylabel("Number genes")
        plt.axvline(x=thres_t, linestyle='--')
        plt.savefig(data_path/'plots'/f'low_gene_filter.png',bbox_inches = "tight",dpi=200)
        plt.close()
    nr_gene_before = len(gene_level)
    gene_level = gene_level[gene_level.above_tres >= nr_patients*0.75]
    gene_counts = gene_counts[gene_counts['sample'].isin(gene_level['sample'])]

    nr_pairs_before = len(gene_mirna_pairs)
    filtered_gene_mirna_pairs = gene_mirna_pairs[gene_mirna_pairs.ensembl_gene_id.isin(gene_level['sample'])]
    logger.info(f'After filter gene expression > {thres_t} for >= 25% of samples: {len(gene_level)}/{nr_gene_before} genes and {len(filtered_gene_mirna_pairs)}/{nr_pairs_before} pairs left.')

    #filter out (gene,miRNA) pairs if not at least 1 nonbinding transcript & 1 binding transcript
    nr_pairs_before = len(gene_mirna_pairs)
    t_level = transcript_counts.groupby('sample').aggregate({'enough_good_samples':'first'})
    filtered_gene_mirna_pairs['nbs'] = filtered_gene_mirna_pairs.nbs.apply(lambda x: [t for t in x if t in t_level.index])
    filtered_gene_mirna_pairs['bs'] = filtered_gene_mirna_pairs.bs.apply(lambda x: [t for t in x if t in t_level.index])
    filtered_gene_mirna_pairs = filtered_gene_mirna_pairs[filtered_gene_mirna_pairs.nbs.apply(lambda x: sum(t_level.loc[x].enough_good_samples)>=1)]
    filtered_gene_mirna_pairs = filtered_gene_mirna_pairs[filtered_gene_mirna_pairs.bs.apply(lambda x: sum(t_level.loc[x].enough_good_samples)>=1)]

    nr_genes_before = len(gene_counts['sample'].unique())
    left_genes = filtered_gene_mirna_pairs.ensembl_gene_id.unique()
    gene_counts = gene_counts[gene_counts['sample'].isin(left_genes)]

    nr_transcript_before = len(transcript_counts['sample'].unique())
    t2g = get_map_g2t(input_path,g2t=False)
    t2g = t2g[t2g.gene_id.isin(left_genes)]
    transcript_counts = transcript_counts[transcript_counts['sample'].isin(t2g.transcript_id)]

    logger.info(f'After filter high expression for at least 1 b & 1 nb transcript/gene: {len(transcript_counts["sample"].unique())}/{nr_transcript_before} transcripts, {len(left_genes)}/{nr_genes_before} genes and {len(filtered_gene_mirna_pairs)}/{nr_pairs_before} pairs left.')
    return filtered_gene_mirna_pairs, transcript_counts, gene_counts

#filter miRNA-gene pairs with existing expression data for alternative splicing and sufficient expression
def calc_gene_mirna_pairs(data_path, nested_path, input_path, path, approach, disease):
    file0 = data_path/'filtered_gene_mirna_pairs.pickle'
    if not file0.is_file():
        logger.info(f'approach {approach}')
        #compare expression with bindingsites & only keep common: 541 miRNAs, 176.332 transcripts, 56.627 genes x 517 samples
        bs_transcript_level, mirna_counts, transcript_counts, gene_counts = get_common_with_bs(nested_path, input_path, disease)
        #from here on depends on approach
        gene_mirna_pairs, gene_counts, transcript_counts, mirna_counts = alternative_splicing_filter(path, input_path, bs_transcript_level, gene_counts, transcript_counts, mirna_counts, approach)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        nr_bs = gene_mirna_pairs.bs.apply(lambda x: len(x)).sum()
        nr_nbs = gene_mirna_pairs.nbs.apply(lambda x: len(x)).sum()
        bar = ax.bar(1,nr_bs)
        bar = ax.bar(2,nr_nbs)
        plt.xticks([1,2], ('binding', 'nonbinding'))
        plt.ylabel("Number transcripts")
        plt.savefig(data_path/'plots'/f'nb_b_hist.png',bbox_inches = "tight",dpi=200)
        plt.close()
        mirna_counts.mirna_count.hist()
        plt.title('Distribution of miRNA expression')
        plt.xlabel("miRNA expression")
        plt.ylabel("Number miRNAs")
        plt.savefig(data_path/'plots'/f'mirna_hist.png',bbox_inches = "tight",dpi=200)
        plt.close()
        mirna_counts.to_pickle(data_path/'filtered_miRNA_counts.pickle')
        transcript_counts = low_transcript_expression_filter(data_path, transcript_counts, do_plots=True)
        transcript_counts.to_pickle(data_path/'temp_transcript_counts.pickle')
        gene_counts.to_pickle(data_path/'temp_gene_counts.pickle')
        gene_mirna_pairs.to_pickle(data_path/'temp_gene_mirna_pairs.pickle')
        filtered_gene_mirna_pairs, transcript_counts, gene_counts = low_gene_expression_filter(input_path, data_path, gene_mirna_pairs, transcript_counts, gene_counts, do_plots=True)
        transcript_counts.to_pickle(data_path/'filtered_transcript_counts.pickle')
        #drop above_thres, enough_good_samples and pivot to have 1 column / transcript
        transcript_counts = transcript_counts.pivot(index=['case_id'],columns='sample',values='transcript_count').reset_index()
        transcript_counts.to_pickle(data_path/'filtered_pivoted_transcript_counts.pickle')
        gene_counts.to_pickle(data_path/'filtered_gene_counts.pickle')
        filtered_gene_mirna_pairs.to_pickle(file0)
    else:
        logger.info(f'loading {approach} filtered_gene_mirna_pairs from pickle file')
        with (open(file0, "rb")) as fp:
            filtered_gene_mirna_pairs = pickle.load(fp)
    return filtered_gene_mirna_pairs

#run to split into separate smaller files for slurm elastic net regression
#n: number pairs we want per file
def split_into_smaller_files(filtered_gene_mirna_pairs, new_path, n):
    nr_files = int(len(filtered_gene_mirna_pairs)/n)+1
    if not new_path.exists():
        os.mkdir(new_path)
        for i in range(nr_files-1):
            filtered_gene_mirna_pairs[i*n:(i+1)*n].to_pickle(new_path/f'gene_mirna_pairs_{i}.pickle')
        i = nr_files-1
        filtered_gene_mirna_pairs[i*n:len(filtered_gene_mirna_pairs)].to_pickle(new_path/f'gene_mirna_pairs_{i}.pickle')
        logger.info(f'Pairs were split into {nr_files} separate files containing each maximum {n} pairs.')
    else:
        logger.info(f'Using existing {nr_files} separate files containing each maximum {n} pairs.')
    return nr_files

# prefilter Pearson correlation of gene-mirna expression
def prefilter(filtered_gene_mirna_pairs, path, data_path, log_path, approach, disease, is_sge, n=10000):
    file0 = data_path/'prefilter_corrs_pairs.pickle'
    if not file0.is_file():
        #split into files for slurm prefilter script
        pair_path = data_path/f'split_pairs_{n}'
        nr_files = split_into_smaller_files(filtered_gene_mirna_pairs, pair_path, n)

        #execute slurm prefilter script
        #squeue, scancel, cat /nfs/proj/lhackl/as/nested/res_prefilter/0.err
        logger.info('Execute prefilter script')
        if is_sge:
            output = subprocess.run(f'qsub -b y -t 1-{nr_files} -o {log_path} -e {log_path} {path}/prefilter_sge.sh {approach} {disease}', capture_output=True, shell=True, universal_newlines=True)
        else:
            output = subprocess.run(['sbatch',f'--array=0-{nr_files-1}%100',f'--output={log_path}/%a.out',f'--error={log_path}/%a.err',f'{path}/prefilter_slurm.sh',f'{approach}',f'{disease}'], capture_output=True, check=True, universal_newlines=True)
            
        logger.info(output.stdout)
        if output.stderr != '': logger.error(output.stderr)
        jobid = output.stdout.split(' ')[2].split('.')[0]

        #read in correlation & put back together
        logger.info('Waiting for prefilter to finish.')
        corrs = [None]*nr_files
        for i in range(0,nr_files):
            file_operations.check_whether_exists(f'{log_path}/prefilter_sge.sh.e{jobid}.{i+1}' if is_sge else data_path/'prefilter_corrs'/f'corrs_{i}.pickle',30,30000,f'{log_path}/{i}.err')
            corrs[i] = pd.read_pickle(data_path/'prefilter_corrs'/f'corrs_{i}.pickle')
        gene_mirna_pairs = pd.concat(corrs)
        gene_mirna_pairs.to_pickle(file0)
        
        logger.info('Read in correlation.')

        #delete unnecessary prefilter_corrs folder
        try:
            shutil.rmtree(data_path/'prefilter_corrs')
        except OSError as e:
            logger.error(f"Error: {e.filename} - {e.strerror}.")

        #delete unnecessary split pair folder
        try:
            shutil.rmtree(pair_path)
        except OSError as e:
            logger.error(f"Error: {e.filename} - {e.strerror}.")

        #plot prefilter correlation
        logger.info(f'Min correlation coefficient: {gene_mirna_pairs["corr"].min()} Max correlation coefficient: {gene_mirna_pairs["corr"].max()}')
        gene_mirna_pairs['corr'].hist()
        plt.ylabel("Absolute number of gene - miRNA pairs")
        plt.xlabel("Pearson correlation coefficient")
        plt.axvline(x = 0, color = 'black', linestyle="--",label='chosen cutoff') 
        plt.legend()
        plt.savefig(data_path/'plots'/f'pearson_corr_coeff_prefilter.png',bbox_inches = "tight",dpi=200)
        plt.close()

        cutoffs = np.linspace(1,-1,17)
        plt.plot(cutoffs, [len(gene_mirna_pairs[gene_mirna_pairs['corr'] < cutoff]) for cutoff in cutoffs])
        plt.ylabel("Number of (gene,miRNA) pairs with coefficient < cutoff")
        plt.xlabel("Cutoff for Pearson correlation coefficient")
        plt.axvline(x = 0, color = 'black', linestyle="--",label='chosen cutoff') 
        plt.legend()
        plt.savefig(data_path/'plots'/f'cutoff_pearson_corr_coeff_prefilter.png',bbox_inches = "tight",dpi=200)
        plt.close()
    else:
        gene_mirna_pairs = pd.read_pickle(file0)

    #keep all negatively correlated pairs
    file = data_path/'filtered_gene_mirna_pairs2.pickle'
    if not file.is_file():
        filtered_gene_mirna_pairs = gene_mirna_pairs[gene_mirna_pairs['corr'] < 0]
        filtered_gene_mirna_pairs['both'] = filtered_gene_mirna_pairs.bs + filtered_gene_mirna_pairs.nbs
        logger.info(f'Keep {len(filtered_gene_mirna_pairs)} pairs ({round((len(filtered_gene_mirna_pairs)/len(gene_mirna_pairs))*100)}%) that are negatively correlated.')
        filtered_gene_mirna_pairs.to_pickle(file)
    else:
        filtered_gene_mirna_pairs = pd.read_pickle(file)
        logger.info('Read in filtered_gene_mirna_pairs from file.')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nr_reduced = filtered_gene_mirna_pairs.nbs.apply(lambda x: len(x)).sum()
    nr_full = filtered_gene_mirna_pairs.both.apply(lambda x: len(x)).sum()
    bar = ax.bar(1,nr_reduced)
    bar = ax.bar(2,nr_full)
    plt.xticks([1,2], ('reduced model','full model'))
    plt.ylabel("Number transcripts")
    plt.text(1 - 0.15, nr_reduced*1.02, f'{nr_reduced:,}')
    plt.text(2 - 0.15, nr_full+(nr_reduced*0.02), f'{nr_full:,}')
    plt.savefig(data_path/'plots'/f'nb_b_hist_after_prefilter.png',bbox_inches = "tight",dpi=200)
    plt.close()
    return filtered_gene_mirna_pairs

#calculate huge automatically dependent on number samples available for disease after filtering
def is_nr_samples_huge(data_path, thres_nr_samples=400):
    samples = pd.read_pickle(data_path/'filtered_miRNA_counts.pickle').case_id.unique()
    return len(samples) > thres_nr_samples

# calculate nested miRNA-gene-level linear regression models for all pairs
def linear_regression(filtered_gene_mirna_pairs, path, data_path, log_path, approach, disease, is_sge, n = 10000, huge=False):
    new_path = data_path/f'reg_split_pairs_{n}'
    nr_files = split_into_smaller_files(filtered_gene_mirna_pairs[['miRNA','ensembl_gene_id','both','nbs']], new_path, n)
    end = nr_files if is_sge else nr_files-1
    jobid = None
    if not os.path.exists(data_path/'nb_t_models'/f'{nr_files-1}_models.pickle'):
        start = 1 if is_sge else 0
        if os.path.exists(data_path/'all_t_models'):
            last_model = 0
            for _,_,files in os.walk(data_path/'all_t_models'):
                for file in files:
                    last_model = max(last_model, int(file.split('_')[0]))
            if last_model != (nr_files-1): #start with model training where execution stopped last time
                start =  last_model+2 if is_sge else last_model+1
                logger.debug(f'Models 0 until {last_model} were already trained, just train remaining models')
        #execute linear regression
        logger.info(f'Run linear regression {start}-{end}')
        if is_sge:
            output = subprocess.run(f'qsub -b y -t {start}-{end} -o {log_path} -e {log_path} {path}/lin_reg_sge.sh {approach} {disease}', capture_output=True, shell=True, universal_newlines=True)
        else:
            output = subprocess.run(['sbatch',f'--array={start}-{end}%100',f'--output={log_path}/%a.out',f'--error={log_path}/%a.err',f'{path}/lin_reg_slurm.sh',f'{approach}',f'{disease}'], capture_output=True, check=True, universal_newlines=True)
        logger.info(output.stdout)
        if output.stderr != '': logger.error(output.stderr)
        jobid = output.stdout.split(' ')[2].split('.')[0]
        logger.info('Waiting for linear regression to finish.')

    if ((not os.path.exists(data_path/'nb_t_models'/f'train_data.pickle')) | (not os.path.exists(data_path/'all_t_models'/f'train_data.pickle'))):
        logger.info('Reading in models.')
        #after finished execution: read in trained models back into one file
        for type_ in [f'nb_t_models',f'all_t_models']:
            models = {}
            train_data = {}
            test_data = {}
            for i in range(nr_files):
                error_file = (f'{log_path}/lin_reg_sge.sh.e{jobid}.{i+1}' if jobid!=None else None) if is_sge else f'{log_path}/{i}.err'
                if not huge:
                    file_operations.check_whether_exists(data_path/type_/f'{i}_models.pickle',180,30000,error_file)
                    models.update(pd.read_pickle(data_path/type_/f'{i}_models.pickle'))
                    file_operations.check_whether_exists(data_path/type_/f'{i}_train_data.pickle',30,30000,error_file)
                    train_data.update(pd.read_pickle(data_path/type_/f'{i}_train_data.pickle'))
                    file_operations.check_whether_exists(data_path/type_/f'{i}_test_data.pickle',30,30000,error_file)
                    test_data.update(pd.read_pickle(data_path/type_/f'{i}_test_data.pickle'))
            if not huge:
                with open(data_path/type_/'models.pickle', 'wb') as handle:
                    pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for file_name in listdir(data_path/type_):
                    if file_name.endswith('_models.pickle'):
                        os.remove(data_path/type_/file_name)
                with open(data_path/type_/'train_data.pickle', 'wb') as handle:
                    pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for file_name in listdir(data_path/type_):
                    if file_name.endswith('_train_data.pickle'):
                        os.remove(data_path/type_/file_name)
                with open(data_path/type_/'test_data.pickle', 'wb') as handle:
                    pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for file_name in listdir(data_path/type_):
                    if file_name.endswith('_test_data.pickle'):
                        os.remove(data_path/type_/file_name)
        logger.info('Read in models.')

        #delete unnecessary split pair folder
        try:
            shutil.rmtree(new_path)
        except OSError as e:
            logging.error(f"Error: {e.filename} - {e.strerror}.")

        #filter models by rmse score
    if huge:
        logger.info('Filter models by RMSE score')
        all_t_models = {}
        for i in range(nr_files):
            all_t_models.update(model_processing.filter_models_by_rmse(data_path,f'all_t_models',prefix=f'{i}_',huge=huge))
        with open(data_path/'all_t_models'/'filtered_models.pickle', 'wb') as handle:
            pickle.dump(all_t_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for file_name in listdir(data_path/'all_t_models'):
            if file_name.endswith('_filtered_models.pickle'):
                os.remove(data_path/'all_t_models'/file_name)
        nb_t_models = {}
        for i in range(nr_files):
            nb_t_models.update(model_processing.filter_models_by_rmse(data_path,f'nb_t_models',prefix=f'{i}_',huge=huge))
        with open(data_path/'nb_t_models'/'filtered_models.pickle', 'wb') as handle:
            pickle.dump(nb_t_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for file_name in listdir(data_path/'all_t_models'):
            if file_name.endswith('_filtered_models.pickle'):
                os.remove(data_path/'all_t_models'/file_name)
        
        #plot
        for type_ in [f'nb_t_models',f'all_t_models']:
            file = data_path/type_/f'before.pickle'
            if not file.is_file():
                before = [None]*nr_files
                for i in range(nr_files):
                    with open(data_path/type_/f'{i}_before.pickle', 'rb') as handle:
                        before[i] = pickle.load(handle)
                before = [item for sublist in before for item in sublist]
                with open(file, 'wb') as handle:
                    pickle.dump(before, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #delete unnecessary before files TODO
                for file_name in os.listdir(data_path/type_):
                    if file_name.endswith("_before.pickle"):
                        os.remove(os.path.join(data_path/type_, file_name))
            else:
                with open(file, 'rb') as handle:
                    before = pickle.load(handle)        
            with open(data_path/type_/f'filtered_models.pickle', 'rb') as handle:
                filtered_models = pickle.load(handle)
            logger.info(f'  Number models before: {len(before)}')
            logger.info(f'  Number models after: {len(filtered_models)}')
            #plot histogram of rooted mean squared error score
            fig1, ax1 = plt.subplots(figsize=(6,4))
            bins_before = np.arange(min(before),max(before),(max(before)-min(before))/100)
            plt.hist(before, bins=bins_before, alpha=0.4, color='tab:blue',label='before')
            plt.hist([rmse_score for (rmse_score, regr, t_pvalues, f_pvalue) in filtered_models.values()], bins=bins_before, alpha=0.5, color='tab:green',label='after')
            plt.ylabel(f"Number (miRNA, gene) {type_}")
            plt.xlabel("Rooted mean squared error")
            plt.axvline(x = 0.7, color = 'black', linestyle="--",label='chosen threshold') 
            plt.legend()
            plt.savefig(data_path/'plots'/f'{type_}_rmse_filter.png',bbox_inches = "tight",dpi=200)
            plt.close()

    else:
        do_plot=True
        all_t_models = model_processing.filter_models_by_rmse(data_path,f'all_t_models',do_plot=do_plot)
        nb_t_models = model_processing.filter_models_by_rmse(data_path,f'nb_t_models',do_plot=do_plot)

    common_pairs = model_processing.get_common_models(data_path, nb_t_models, all_t_models)
    log_lls = model_processing.calc_ll(data_path, common_pairs, do_plot=True)
    logger.info(f'statistically significant p-values (<0.05) for {len(log_lls[log_lls.corr_p_val < 0.05])}/{len(log_lls)} models ({len(log_lls[log_lls.corr_p_val < 0.05])/len(log_lls)}%)')

def main(argv):
    # parsing parameters
    path = os.path.dirname(os.path.realpath(__file__))
    disease = None
    approach = None
    workload_manager = None
    logging.basicConfig(filename=f'{path}/fallback.log', filemode='a', force=True, format='%(asctime)s %(levelname)s - %(message)s')
    try:
        opts, args = getopt.getopt(argv, 'd:a:w:')
    except getopt.GetoptError:
        logger.error('Error: Incorrect argument. Usage: python nested.py -d <disease> -a <approach> -w <workload_manager>')
        sys.exit(2)
    for i in opts:
        if i[0] == '-d':
            disease = i[1]
        if i[0] == '-a':
            approach = i[1]
        if i[0] == '-w':
            workload_manager = i[1]
    if ((disease == None)|(approach == None)|(workload_manager == None)):
        logger.error('Error: Missing argument. Usage: python nested.py -d <disease> -a <approach> -w <workload_manager>')
        sys.exit(2)
    if not approach in ['A','B','C']:
        logger.error('Error: Wrong argument. -a <approach> has to be in [A,B,C]')
        sys.exit(2)
    if not workload_manager in ['sge','slurm']:
        logger.error('Error: Wrong argument. -w <workload_manager> has to be sge or slurm')
        sys.exit(2)
    else: 
        is_sge = (workload_manager == 'sge')
    input_path = Path(path/'input_data')
    nested_path = Path(path/disease)
    if not os.path.exists(nested_path):
        os.makedirs(nested_path)
        os.makedirs(nested_path/'plots')
    data_path = Path(nested_path/approach)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        os.makedirs(data_path/'plots')
    prefilter_log_path = Path(path/'res_prefilter')
    if not os.path.exists(prefilter_log_path):
        os.makedirs(prefilter_log_path)
    lin_reg_log_path = Path(path/'res_linear')
    if not os.path.exists(lin_reg_log_path):
        os.makedirs(lin_reg_log_path)

    if os.path.exists(f'{data_path}/nested_{disease}_{approach}.log'):
        logging.basicConfig(filename=f'{data_path}/nested_{disease}_{approach}.log', filemode='a', force=True, format='%(asctime)s %(levelname)s - %(message)s')
    else:
        logging.basicConfig(filename=f'{data_path}/nested_{disease}_{approach}.log', filemode='w', force=True, format='%(asctime)s %(levelname)s - %(message)s')
    filtered_gene_mirna_pairs = calc_gene_mirna_pairs(data_path, nested_path, input_path, path, approach, disease)          
    filtered_gene_mirna_pairs = prefilter(filtered_gene_mirna_pairs, path, data_path, prefilter_log_path, approach, disease, is_sge=is_sge)
    huge = is_nr_samples_huge(data_path)
    linear_regression(filtered_gene_mirna_pairs, path, data_path, lin_reg_log_path, approach, disease, is_sge=is_sge, huge=huge)
    logger.info('Done.')

if __name__ == '__main__':
    main(sys.argv[1:])