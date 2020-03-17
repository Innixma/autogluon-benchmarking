import pandas as pd

from ..constants import *


def clean_result(result_df, folds_to_keep=None, remove_invalid=True):
    if folds_to_keep is None:
        folds_to_keep = sorted(list(result_df[FOLD].unique()))
    folds_required = len(folds_to_keep)
    result_df = result_df[result_df[FOLD].isin(folds_to_keep)]
    datasets = list(result_df[DATASET].unique())
    model_names = list(result_df[FRAMEWORK].unique())
    result_clean = []
    for model_name in model_names:
        for dataset in datasets:
            fd_result = result_df[(result_df[FRAMEWORK] == model_name) & (result_df[DATASET] == dataset)].copy()
            fd_result = fd_result[fd_result[METRIC_ERROR].notnull()]
            if remove_invalid:
                if len(fd_result) == folds_required:
                    result_clean.append(fd_result)
            else:
                result_clean.append(fd_result)
    if len(result_clean) == 0:
        results_clean_df = pd.DataFrame()
    else:
        results_clean_df = pd.concat(result_clean, ignore_index=True)

    return results_clean_df
