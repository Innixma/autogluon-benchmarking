import numpy as np
import pandas as pd

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS

from . import preprocess_utils
from ..constants import *


def preprocess_openml_input(path, framework_suffix=None, framework_rename_dict=None, folds_to_keep=None):
    raw_input = load_pd.load(path)
    raw_input = _rename_openml_columns(raw_input)
    if framework_rename_dict is not None:
        for key in framework_rename_dict.keys():
            raw_input[FRAMEWORK] = [framework_rename_dict[key] if framework[0] == key else framework[0] for framework in zip(raw_input[FRAMEWORK])]
    if framework_suffix is not None:
        raw_input[FRAMEWORK] = [framework[0] + framework_suffix for framework in zip(raw_input[FRAMEWORK])]

    with_prob_type_input = _infer_problem_type(raw_input)
    fixed_input = _fix_results(with_prob_type_input)
    fixed_input[METRIC_ERROR] = [1 - result if ptype == BINARY else result for result, ptype in zip(fixed_input[METRIC_SCORE], fixed_input[PROBLEM_TYPE])]
    cleaned_input = preprocess_utils.clean_result(fixed_input, folds_to_keep=folds_to_keep, remove_invalid=False)
    return cleaned_input


def _fix_results(results):
    results = results.copy()
    results = _fix_auc_errors(results)
    results = _fix_incorrect_score_errors(results)
    return results


def _rename_openml_columns(result_df):
    renamed_df = result_df.rename(columns={
        TASK: DATASET,
        FRAMEWORK: FRAMEWORK,
        RESULT: METRIC_SCORE,
        DURATION: TIME_TRAIN_S,
        PREDICT_DURATION: TIME_INFER_S,
    })
    return renamed_df


# Happens in nomao for autogluon, value is flipped, occurs due to weird class names in binary such as '1' = True, '2' = False in nomao
# Occurs for credit-g, blood-transfusion, nomao
# Note, it is possible a model could be actively worse than random but I doubt that would happen or that they could be significantly worse than random for this to help them much.
def _fix_auc_errors(result_df):
    new_result = result_df.copy()
    new_result[METRIC_SCORE] = [_fix_row(result, problem_type) for result, problem_type in zip(new_result[METRIC_SCORE], new_result[PROBLEM_TYPE])]
    if 'auc' in new_result.columns:
        new_result['auc'] = [auc if (problem_type != BINARY) else result for auc, result, problem_type in zip(new_result['auc'], new_result[METRIC_SCORE], new_result[PROBLEM_TYPE])]

    return new_result


# TODO: Assumes AUC metric, also works for Accuracy
def _fix_row(score, problem_type):
    if problem_type != BINARY:
        return score
    elif score > 1:
        return np.nan
    elif score < 0:
        return np.nan
    elif score < 0.5:
        return 1 - score
    else:
        return score


# OpenML has an error due to column shift in their report, this fixes that.
def _fix_incorrect_score_errors(result_df):
    new_result = result_df.copy()
    if 'auc' in new_result.columns:
        new_result['auc'] = [auc if (problem_type == BINARY) else np.nan for auc, problem_type in zip(new_result['auc'], new_result[PROBLEM_TYPE])]
    if 'logloss' in new_result.columns:
        new_result['logloss'] = [result if (problem_type == MULTICLASS) else np.nan for logloss, result, problem_type in zip(new_result['logloss'], new_result[METRIC_SCORE], new_result[PROBLEM_TYPE])]
    return new_result


def _infer_problem_type(result_df):
    if PROBLEM_TYPE not in result_df.columns:
        datasets = list(result_df[DATASET].unique())
        result_with_problem_type = []
        for dataset in datasets:
            fd_result = result_df[result_df[DATASET] == dataset].copy()
            if 'auc' not in fd_result.columns:
                fd_result[PROBLEM_TYPE] = BINARY  # TODO: Not correct
            elif fd_result['auc'].notnull().values.any():
                fd_result[PROBLEM_TYPE] = BINARY
            else:
                fd_result[PROBLEM_TYPE] = MULTICLASS
            result_with_problem_type.append(fd_result)
        if len(result_with_problem_type) == 0:
            results_with_problem_type_df = pd.DataFrame()
        else:
            results_with_problem_type_df = pd.concat(result_with_problem_type, ignore_index=True)
        return results_with_problem_type_df
    else:
        return result_df
