from autogluon.utils.tabular.utils.loaders import load_pd

from . import preprocess_utils
from ..constants import *


def preprocess_kaggle_input(path, framework_suffix=None, framework_rename_dict=None):
    raw_input = load_pd.load(path)

    raw_input = _rename_kaggle_input(raw_input)
    raw_input[FOLD] = 0
    if METRIC_SCORE not in raw_input.columns:
        raw_input[METRIC_SCORE] = -raw_input[METRIC_ERROR]

    if framework_rename_dict is not None:
        for key in framework_rename_dict.keys():
            raw_input[FRAMEWORK] = [framework_rename_dict[key] if framework[0] == key else framework[0] for framework in zip(raw_input[FRAMEWORK])]
    if framework_suffix is not None:
        raw_input[FRAMEWORK] = [framework[0] + framework_suffix for framework in zip(raw_input[FRAMEWORK])]
    cleaned_input = preprocess_utils.clean_result(raw_input, folds_to_keep=[0])
    return cleaned_input


def _rename_kaggle_input(result_df):
    renamed_df = result_df.rename(columns={
        'DATASET': DATASET,
        'MODEL_NAME': FRAMEWORK,
        'METRIC_SCORE': METRIC_SCORE,
        'METRIC_ERROR': METRIC_ERROR,
        'TIME_TRAIN_S': TIME_TRAIN_S,
        'TIME_INFER_S': TIME_INFER_S,
        'PROBLEM_TYPE': PROBLEM_TYPE,
    })
    return renamed_df
