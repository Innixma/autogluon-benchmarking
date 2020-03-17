#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoML performance evaluation in Kaggle"""


import os, pickle, json, time, traceback, tempfile, numbers
import pprint as pp
import boto3
import fire
import numpy as np
import pandas as pd
from pandas import DataFrame
from copy import deepcopy
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Tuple, Mapping, List

from autogluon.utils.tabular.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS

from autogluon_utils.benchmarking.kaggle.fetch_datasets import PRE_PROCESSED_PARQUET_TRAIN_FILE_NAME, PRE_PROCESSED_PARQUET_TEST_FILE_NAME, \
    PRE_PROCESSED_TRAIN_FILE_NAME, PRE_PROCESSED_TEST_FILE_NAME, get_bucket_and_key
from autogluon_utils.benchmarking.kaggle.postprocessors.kaggle_postprocessor import KaggleSubmissionPostProcessor
from autogluon_utils.configs.kaggle.kaggle_competitions import *
from autogluon_utils.configs.kaggle.common import *
from autogluon_utils.configs.kaggle.constants import *
from autogluon_utils.kaggle_tools.kaggle_utils import kaggle_api, submit_kaggle_competition

def evaluate(fitting_profile=PROFILE_FULL, use_data_in_parquet=False, competitions=KAGGLE_COMPETITIONS,
             predictors=PREDICTORS, tag = TAG):
    api = kaggle_api()
    print(f'Using benchmarking profile: {fitting_profile}')
    pp.pprint(FITTING_PROFILES[fitting_profile])

    trial_meta = {  # sub-dirs will be formed using trials_meta data
        TIMESTAMP_STR: datetime.utcnow().strftime('%Y_%m_%d-%H_%M_%S'),
        PROFILE: fitting_profile,
    }

    for predictor_name, predictor_type in predictors.items():
        print(f'\n\n################################################################################')
        print(f'\nEvaluating predictor: {predictor_name}\n')
        trial_meta[PREDICTOR] = predictor_name

        for competition_meta in competitions:
            metrics = {}
            predictor = predictor_type()
            profile = deepcopy(FITTING_PROFILES[fitting_profile])
            print(f'================================================================================')
            print(f'Running competition: {competition_meta[NAME]}')
            print(f'================================================================================')
            pp.pprint(competition_meta)
            s3_path = competition_s3_path(competition_meta[NAME])
            output_path = os.path.join(s3_path, 'outputs', trial_meta[PREDICTOR], trial_meta[PROFILE], 
                                       tag, trial_meta[TIMESTAMP_STR])
            print("All results will be stored in: %s" % output_path)
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    train_data, test_data = load_datasets(s3_path, competition_meta, profile, use_data_in_parquet, metrics)
                    y_preds, class_order = predictor.run_competition(train_data, test_data, competition_meta, profile, metrics, tmp_dir)
                    print("Saving trained models to s3...")
                    _upload_files(tmp_dir+"/", output_path)
                    
                    submission = apply_post_processing(y_preds, class_order, test_data, competition_meta)
                    local_submission_file = os.path.join(tmp_dir, 'submission.csv')
                    submission.to_csv(local_submission_file, index=False)
                    print("Saving submission file to s3...")
                    _upload_files(tmp_dir+"/", output_path)
                    
                    print(f'================================================================================')
                    print(f'Submitting predictions to kaggle to score performance...')
                    submission_result = submit_kaggle_competition(competition_meta[NAME], local_submission_file)
                    metrics['public_score'] = submission_result.public_score
                    metrics['private_score'] = submission_result.private_score
                    metrics['error_description'] = submission_result.error_description
                    metrics['date'] = submission_result.date
                    metrics['rank'] = submission_result.leaderboard_rank
                    metrics['num_teams'] = submission_result.num_teams
                    metrics['metric_error'] = get_metric_error(submission_result.private_score, 
                                                submission_result.public_score, competition_meta)
                    print(metrics)
                    with open(os.path.join(tmp_dir, 'kaggle_submission_result.json'), 'w') as f:
                        json.dump(submission_result, f, default=str)
                    print(f'================================================================================')
                    
                    print("Results:")
                    pp.pprint(metrics)
                    metrics_file = os.path.join(tmp_dir, 'metrics.json')
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, default=str)
                    
                    print("Saving final results to s3...")
                    _upload_files(tmp_dir + '/', output_path)
            except Exception as e:
                print('!!! Exception !!!')
                print(e)
                traceback.print_exc()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with open(os.path.join(tmp_dir, 'exception.txt'), 'w') as f:
                        traceback.print_exc(file=f)

                    _upload_files(tmp_dir + '/', os.path.join(output_path, 'error'))
                    print("See file for exception details: %s" % os.path.join(output_path, 'error', 'exception.txt'))

def get_metric_error(private_score, public_score, competition_meta):
    metrics = CLASSIFICATION_METRICS.copy()
    metrics.update(REGRESSION_METRICS)
    if isinstance(private_score, numbers.Number):
        score = private_score
    elif isinstance(public_score, numbers.Number):
        score = public_score
    else:
        return None
    if competition_meta[EVAL_METRIC] not in metrics:
        raise ValueError("unknown metric specified: %s" % competition_meta[EVAL_METRIC])
    if metrics[competition_meta[EVAL_METRIC]]._sign == 1:
        return 1 - score  # flip applied to AUC, Gini, R^2 etc.
    else:
        return score

def _upload_files(src_path, dest_path):
    s3_client = boto3.client('s3')
    bucket, key_prefix = get_bucket_and_key(dest_path)
    for subdir, dirs, files in os.walk(src_path):
        for file in files:
            local_file_path = os.path.join(subdir, file)
            key = os.path.join(key_prefix, local_file_path[len(src_path):])
            s3_client.upload_file(local_file_path, bucket, key)
            print(f'    - {local_file_path} -> s3://{bucket}/{key}')


def apply_post_processing(y_predproba: np.array, class_order : List[str], test_data: task.Dataset, competition_meta) -> DataFrame:
    print('Applying predictions post-processing to generate submission...')
    post_processor: KaggleSubmissionPostProcessor = competition_meta[POST_PROCESSOR]
    submission = post_processor.postprocess(class_order, test_data, y_predproba, competition_meta)
    return submission

def load_datasets(competition_s3_path: str, competition_meta: Mapping[str, str], profile: dict,
                  use_data_in_parquet: bool, metrics: Mapping[str, any]) -> Tuple[DataFrame, DataFrame]:
    print('Loading datasets for fitting...')
    output_prefix = '/'.join(['s3:/', competition_s3_path, 'processed'])

    if use_data_in_parquet:
        train_path = f'{output_prefix}/{PRE_PROCESSED_PARQUET_TRAIN_FILE_NAME}'
        df_train = pd.read_parquet(train_path)
    else:
        train_path = f'{output_prefix}/{PRE_PROCESSED_TRAIN_FILE_NAME}'
        df_train = pd.read_csv(train_path, index_col=competition_meta[INDEX_COL], low_memory=False)
    metrics['num_train_rows'] = df_train.shape[0]
    metrics['num_train_cols'] = df_train.shape[1]
    print(f' - Loaded training set: shape {df_train.shape}')

    if use_data_in_parquet:
        test_path = f'{output_prefix}/{PRE_PROCESSED_PARQUET_TEST_FILE_NAME}'
        df_test = pd.read_parquet(test_path)
    else:
        test_path = f'{output_prefix}/{PRE_PROCESSED_TEST_FILE_NAME}'
        df_test = pd.read_csv(test_path, index_col=competition_meta[INDEX_COL], low_memory=False)
    metrics['num_test_rows'] = df_test.shape[0]
    metrics['num_test_cols'] = df_test.shape[1]
    print(f' - Loaded test set: shape {df_test.shape}')

    subsample = profile[SUBSAMPLE] if SUBSAMPLE in profile else None
    if subsample is not None:
        # del profile[SUBSAMPLE]
        print(f' - Using training-data subsample of size: {subsample}')
        df_train = df_train.head(subsample)

    # Sanity check: confirm that dtypes are the same between train and test
    print('Checking train vs test data types mismatch...')
    train_vs_test_types_issues_found = False
    for col, tst_type in df_test.dtypes.items():
        trn_type = df_train[col].dtype
        if trn_type != tst_type:
            train_vs_test_types_issues_found = True
            print(f' - WARNING: column {col} has different dtypes in train ({trn_type}) and test ({tst_type}) - this might reduce results quality')
    if not train_vs_test_types_issues_found:
        print(' - No train vs test data types mismatches found')

    # train_data = task.Dataset(df=df_train, subsample=subsample) # these are autogluon-specific
    # test_data = task.Dataset(df=df_test, subsample=subsample)
    # return train_data, test_data
    return df_train, df_test

def run(competition_name, fitting_profile, predictor, tag = TAG):
    competition = [c for c in KAGGLE_COMPETITIONS if c[NAME] == competition_name]
    _predictor = {k: v for k, v in PREDICTORS.items() if k == predictor}
    print(f'Running benchmark for {competition} | profile {fitting_profile} | predictor {_predictor} | tag {tag}')
    evaluate(fitting_profile=fitting_profile, use_data_in_parquet=False, competitions=competition,
             predictors=_predictor, tag=tag)
    print(f'Finished benchmark for {competition} | profile {fitting_profile} | predictor {_predictor} | tag {tag}')


if __name__ == '__main__':
    fire.Fire(run)  # CLI wrapper

# To run from bash:
# export PYTHONPATH='/home/ubuntu/autogluon/:/home/ubuntu/autogluon-utils'
# python -u /home/ubuntu/autogluon-utils/autogluon_utils/benchmarking/kaggle/evaluate_results.py $COMPETITION $PROFILE $PREDICTOR 


"""

# This is all old AutoGluon-specific functionality:

def upload_results(tmp_dir, output_path, submission):
    print('Uploading models...')
    models_local_path = os.path.join(tmp_dir, 'models', '')
    models_s3_path = os.path.join(output_path, 'models')
    _upload_files(models_local_path, models_s3_path)

def log_metrics(tmp_dir: TemporaryDirectory, output_path: str, fit_summary: Mapping[str, any], leaderboard: DataFrame, metrics: Mapping[str, any]):
    metrics_local_path = '/'.join([tmp_dir, 'metrics/'])
    if not os.path.exists(os.path.dirname(metrics_local_path)):
        os.makedirs(os.path.dirname(metrics_local_path))

    pp.pprint(fit_summary)
    fit_summary_str: Mapping[str, str] = _convert_dict_values_to_str(fit_summary)
    with open(metrics_local_path + 'fit_summary.json', 'w') as f:
        json.dump(fit_summary_str, f)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(leaderboard)
    leaderboard.to_csv(metrics_local_path + 'leaderboard.csv', index=False)

    metrics['models.trained'] = len(leaderboard)
    pp.pprint(metrics)
    with open(metrics_local_path + 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    print('Uploading metrics...')
    models_local_path = '/'.join([tmp_dir, 'metrics/'])
    models_s3_path = '/'.join([output_path, 'metrics'])
    _upload_files(models_local_path, models_s3_path)

def predict(predictor: Predictor, test_data: DataFrame, metrics: Mapping[str, any]):
    print('Running prediction...')
    ts = time.time()
    y_predproba = predictor.predict_proba(test_data)
    metrics['predict_time'] = (time.time() - ts) * 1000
    return y_predproba


def fit_models(predictor: Predictor, train_data: task.Dataset, competition_meta: Mapping[str, str], profile: Mapping[str, any],
               metrics: Mapping[str, any], tmp_dir: TemporaryDirectory) -> Tuple[Mapping[str, any], DataFrame]:
    print(f'Fitting models...')

    ts = time.time()
    predictor.fit(train_data, competition_meta, profile, tmp_dir)
    metrics['fit_time'] = (time.time() - ts) * 1000

    fit_summary = predictor.get_fit_summary()
    leaderboard = predictor.get_leaderboard()
    return fit_summary, leaderboard

"""
