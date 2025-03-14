import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
import spm1d
from itertools import combinations
idx = pd.IndexSlice

def get_dataset():
    """
    Load the dataset and perfrom some basic processing 
    - time normalization
    - calculate mean step

    Returns:
    - data: DataFrame containing the dataset
    """
    # check if filepath exists
    filepath = 'data/'
    if not os.path.exists(filepath):
        raise FileNotFoundError('Data Directory not found - ensure it is in the correct location "data/"')

    # load the dataset
    files = os.listdir(filepath)

    data = pd.DataFrame(dtype=float)

    # loop through files
    for file in tqdm(files):
        subject, speed = file.split('_')
        speed = int(speed.split('.')[0])

        if not file.endswith('.csv'):
            continue
        
        # load the data
        df = pd.read_csv(filepath + file, index_col=0, header=[0,1,2,3,4])

        # time normalization    
        df = df.apply(lambda x: resample_timeseries(x, n_samples=101, smooth=True))

        # calculate mean step - per side
        df = df.T.groupby(level=[0,2,3,4]).mean().T
        # pool left and right
        df = df.T.groupby(level=[1,2,3]).mean().T

        # add subject and speed as additional column levels
        new_index_tuples = [(subject, speed) + col for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(new_index_tuples, 
                                               names=['subject', 'speed', 'metric', 'joint', 'dimension'])    
        
        # add to data
        data = pd.concat([data, df], axis=1)

    return data


def get_subject_data(subjects):
    """
    Get the data for the specified subjects.
    All steps for each of the 6 speed conditions are loaded and time normalized.
    if more than 1 subject is specified: mean step is calculated, else all steps are returned.
    
    Args: subjects - list of subject ids
    Returns: data - DataFrame containing the data for the specified subjects
    """
    # check if filepath exists
    filepath = 'data/'
    if not os.path.exists(filepath):
        raise FileNotFoundError('Data Directory not found - ensure it is in the correct location "data/"')
    
    # more than 1 subjects --> calculate mean
    if len(subjects) > 1:
        print('Calculating mean step for multiple subjects')
        mean_only = True
    else:
        mean_only = False

    # load data for each subject
    data = pd.DataFrame(dtype=float)
    for subject in subjects:
        print(f'Loading data for subject {subject}')
        # get files for subject
        files = [x for x in os.listdir(filepath) if x.startswith(subject)]
        if len(files) == 0:
            raise FileNotFoundError(f'No files found for subject {subject}')
        
        # loop through files
        for file in files:
            subject, speed = file.split('_')
            speed = int(speed.split('.')[0])

            if not file.endswith('.csv'):
                continue
            
            # load the data
            df = pd.read_csv(filepath + file, index_col=0, header=[0,1,2,3,4])

            # time normalization    
            df = df.apply(lambda x: resample_timeseries(x, n_samples=101, smooth=True))

            # calculate mean step - per side if more than 1 subject
            if mean_only:
                # calculate mean step - per side
                df = df.T.groupby(level=[0,2,3,4]).mean().T
                # pool left and right
                df = df.T.groupby(level=[1,2,3]).mean().T

                # add subject and speed as additional column levels
                new_index_tuples = [(subject, speed) + col for col in df.columns]
                df.columns = pd.MultiIndex.from_tuples(new_index_tuples, 
                                               names=['subject', 'speed', 'metric', 'joint', 'dimension']) 
            else:
                # add subject and speed as additional column levels
                new_index_tuples = [(subject, speed) + col for col in df.columns]
                df.columns = pd.MultiIndex.from_tuples(new_index_tuples, 
                                                names=['subject', 'speed', 'side', 'step', 'metric', 'joint', 'dimension'])
            
            # add to data
            data = pd.concat([data, df], axis=1)

    return data



def resample_timeseries(ts, n_samples=201, smooth=False):
    """
    Resample a time series to a fixed number of samples using linear interpolation.
    Optionally, apply a Gaussian filter to smooth the resampled time series."""
    ts = ts.dropna()
    if len(ts) == 0:
        return pd.Series(np.tile(np.nan, n_samples), name=ts.name)
    ts_resampled = resample(ts, n_samples)
    if smooth:
        ts_smoothened = gaussian_filter1d(ts_resampled, sigma=4, mode='nearest')  # original: sigma=4, no mode specified
    else:
        ts_smoothened = ts_resampled
    return ts_smoothened


def rm_anova(data_cond, speeds):
    """
    Perform repeated measures ANOVA on the data for a given condition.
    """
    # get test values
    y = [data_cond.loc[idx[:,speed],:].values for speed in speeds]
    Y = np.vstack(y)
    # speed labels
    A = np.repeat(speeds, data_cond.loc[idx[:,speeds[0]],:].values.shape[0])
    # subject labels
    SUBJ = np.tile(np.arange(data_cond.loc[idx[:,speeds[0]],:].values.shape[0]), len(speeds))

    # perform RM ANOVA
    alpha        = 0.05
    equal_var    = True
    F            = spm1d.stats.anova1rm(Y, A, SUBJ, equal_var)
    Fi           = F.inference(alpha)

    # Get significant regions for ANOVA
    anova_sig_regions = Fi.clusters if Fi.h0reject else []
    # Get ANOVA p-value
    anova_p_value = Fi.p  
    
    return {"regions": anova_sig_regions, "pval": anova_p_value, "F": F, "Fi": Fi}


def post_hoc(data_cond, speeds):
    """
    Perform post-hoc paired t-tests on the data for a given condition for all speed pairs.
    """
    # Get all 2-speed combinations for pairwise comparisons
    speed_pairs = list(combinations(speeds, 2)) 

    # Bonferroni Correction for alpha
    alpha = 0.05/len(speed_pairs)

    post_hoc_results = {}

    # pairwise post-hoc comparisons
    for s1, s2 in speed_pairs:
        y1 = data_cond.loc[idx[:,s1],:].values
        y2 = data_cond.loc[idx[:,s2],:].values

        t = spm1d.stats.ttest_paired(y1, y2)
        ti = t.inference(alpha=alpha, two_tailed=True, interp=True)

        post_hoc_results[(s1, s2)] = {"regions": ti.clusters if ti.h0reject else [], 
                                      "pval": ti.p, "t": t, "ti": ti}
        
    return post_hoc_results
   