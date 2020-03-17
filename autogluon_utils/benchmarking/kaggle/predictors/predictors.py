import os, time, json, pandas
import pprint as pp
from typing import List, Mapping
import numpy as np
import pandas as pd
from pandas import DataFrame
from tempfile import TemporaryDirectory
from datetime import datetime

from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction import TabularPredictor

from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon_utils.configs.kaggle.constants import *
from autogluon_utils.benchmarking.baselines.h2o_base.h2o_base import H2OBaseline
from autogluon_utils.benchmarking.baselines.tpot_base.tpot_base import TPOTBaseline
from autogluon_utils.benchmarking.baselines.autosklearn_base.autosklearn_base import AutoSklearnBaseline
from autogluon_utils.benchmarking.baselines.autoweka.methods_autoweka import autoweka_fit_predict
from autogluon_utils.benchmarking.baselines.gcp.methods_gcp import gcptables_fit_predict

PREDICT_PROBA_METRICS = ['roc_auc', 'log_loss'] # only metrics where we must predict probabilities

class Predictor:

    def __init__(self):
        self.predictor = None

    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str],
                        profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        """ Returns tuple:
            (y_preds, output_files, class_order)
            y_preds : numpy array
                Contains predicted classes unless metric calls for probabilistic predictions.
                If metric is 'roc_auc', just a 1D numpy array of positive-class probabilities.
                Otherwise, for classification with probability-based metric, will be numpy array of num_examples x num_classes.
            output_files : 
                List of local paths to files/folders containing miscellaneous useful information. If folder, should not end in slash
            class_order : 
                None if Regression, 
                ordered list of classes corresopnding to columns of y_predproba for Multiclass classification,
                single string indicating the positive class for Binary classification.
            
        """
        raise NotImplementedError()


class AutoGluonPredictor(Predictor):
    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str], 
                         profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        train_data = task.Dataset(df=train_data)
        test_data = task.Dataset(df=test_data)
        extra_argnames = ['hyperparameter_tune', 'auto_stack', 'verbosity']
        extra_args = {'output_directory': tmp_dir}
        for argname in extra_argnames:
            if argname in profile:
                extra_args[argname] = profile[argname]
        if 'runtime_hr' in profile:
            extra_args['time_limits'] = profile['runtime_hr'] * 3600
        
        print("Fitting AutoGluon with args:")
        print(extra_args)
        t0 = time.time()
        self.predictor = task.fit(
            train_data=train_data,
            label=competition_meta[LABEL_COLUMN],
            eval_metric=competition_meta[EVAL_METRIC],
            problem_type=competition_meta[PROBLEM_TYPE],
            **extra_args
        )
        t1 = time.time()
        metrics['fit_time'] = t1 - t0
        fit_summary = self.predictor.fit_summary(verbosity=1)
        leaderboard = self.predictor.leaderboard()
        pp.pprint(fit_summary)
        fit_summary_str = _convert_dict_values_to_str(fit_summary)
        with open(tmp_dir + 'fit_summary.json', 'w') as f:
            json.dump(fit_summary_str, f)
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(leaderboard)
        leaderboard.to_csv(tmp_dir + 'leaderboard.csv', index=False)
        # output_files = [tmp_dir]
        metrics['num_models_trained'] = len(leaderboard)
        metrics['num_models_ensemble'] = None # Not stored for now
        print("Using AutoGluon for prediction...")
        t2 = time.time()
        if competition_meta[EVAL_METRIC] in PREDICT_PROBA_METRICS:
            y_preds = self.predictor.predict_proba(test_data)
        else:
            y_preds = self.predictor.predict(test_data)
        t3 = time.time()
        metrics['pred_time'] = t3 - t2
        class_order = None
        if competition_meta[PROBLEM_TYPE] == MULTICLASS:
            class_order = self.predictor.class_labels
        elif competition_meta[EVAL_METRIC] == 'roc_auc':
            class_order = self.predictor._learner.label_cleaner.cat_mappings_dependent_var[1]
        return y_preds, class_order # , output_files


class h2oPredictor(Predictor):
    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str], 
                         profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        runtime_sec = None
        if 'runtime_hr' in profile:
            runtime_sec = int(profile['runtime_hr'] * 3600)
        
        print('Fitting h2o...')
        h2o_model = H2OBaseline()
        
        num_models_trained, num_models_ensemble, fit_time = h2o_model.fit(train_data=train_data, output_directory=tmp_dir,
            label_column=competition_meta[LABEL_COLUMN], problem_type=competition_meta[PROBLEM_TYPE], 
            eval_metric=competition_meta[EVAL_METRIC], runtime_sec=runtime_sec)
        
        print('Predicting with h2o...')
        y_preds, y_prob, predict_time = h2o_model.predict(test_data, pred_class_and_proba=True) # no extra time cost to predict both
        if competition_meta[EVAL_METRIC] in PREDICT_PROBA_METRICS:
            y_preds = y_prob
        
        metrics['num_models_trained'] = num_models_trained
        metrics['num_models_ensemble'] = num_models_ensemble
        metrics['fit_time'] = fit_time
        metrics['pred_time'] = predict_time
        return y_preds, h2o_model.classes


class autosklearnPredictor(Predictor):
    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str], 
                         profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        runtime_sec = None
        if 'runtime_hr' in profile:
            runtime_sec = int(profile['runtime_hr'] * 3600)
        
        print('Fitting autosklearn with runtime_sec=%s...' % runtime_sec)
        autosk = AutoSklearnBaseline()
        num_models_trained, num_models_ensemble, fit_time = autosk.fit(train_data=train_data, output_directory=tmp_dir,
            label_column=competition_meta[LABEL_COLUMN], problem_type=competition_meta[PROBLEM_TYPE], 
            eval_metric=competition_meta[EVAL_METRIC], runtime_sec=runtime_sec)
        
        print('Predicting with autosklearn...')
        predict_proba = (competition_meta[EVAL_METRIC] in PREDICT_PROBA_METRICS)
        y_preds, y_prob, predict_time = autosk.predict(test_data, predict_proba=predict_proba)
        if predict_proba:
            y_preds = y_prob
        
        metrics['num_models_trained'] = num_models_trained
        metrics['num_models_ensemble'] = num_models_ensemble
        metrics['fit_time'] = fit_time
        metrics['pred_time'] = predict_time
        return y_preds, autosk.classes


class tpotPredictor(Predictor):
    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str], 
                         profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        runtime_sec = None
        if 'runtime_hr' in profile:
            runtime_sec = int(profile['runtime_hr'] * 3600)
        
        print('Fitting TPOT...')
        tpot_model = TPOTBaseline()
        
        num_models_trained, num_models_ensemble, fit_time = tpot_model.fit(train_data=train_data, output_directory=tmp_dir,
            label_column=competition_meta[LABEL_COLUMN], problem_type=competition_meta[PROBLEM_TYPE], 
            eval_metric=competition_meta[EVAL_METRIC], runtime_sec=runtime_sec)
        
        print('Predicting with TPOT...')
        predict_proba = (competition_meta[EVAL_METRIC] in PREDICT_PROBA_METRICS)
        y_preds, y_prob, predict_time = tpot_model.predict(test_data, predict_proba=predict_proba)
        if predict_proba:
            y_preds = y_prob
        metrics['num_models_trained'] = num_models_trained
        metrics['num_models_ensemble'] = num_models_ensemble
        metrics['fit_time'] = fit_time
        metrics['pred_time'] = predict_time
        return y_preds, tpot_model.classes


class autowekaPredictor(Predictor):
    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str], 
                         profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        runtime_sec = None
        if 'runtime_hr' in profile:
            runtime_sec = int(profile['runtime_hr'] * 3600)
        
        print('Fitting Auto-WEKA...')
        (num_models_trained, num_models_ensemble, fit_time, y_preds, y_prob, 
         predict_time, class_order) = autoweka_fit_predict(train_data=train_data, test_data=test_data, 
            label_column=competition_meta[LABEL_COLUMN], problem_type=competition_meta[PROBLEM_TYPE], 
            output_directory=tmp_dir, autoweka_path=AUTOWEKA_PATH,
            eval_metric=competition_meta[EVAL_METRIC], runtime_sec=runtime_sec)
        
        if competition_meta[EVAL_METRIC] in PREDICT_PROBA_METRICS:
            y_preds = y_prob
        
        metrics['num_models_trained'] = num_models_trained
        metrics['num_models_ensemble'] = num_models_ensemble
        metrics['fit_time'] = fit_time
        metrics['pred_time'] = predict_time
        return y_preds, class_order


class gcpPredictor(Predictor):
    def run_competition(self, train_data: DataFrame, test_data: DataFrame, competition_meta: Mapping[str, str], 
                         profile: Mapping[str, any], metrics: Mapping[str, any], tmp_dir: TemporaryDirectory):
        gcp_info = {
            'COMPUTE_REGION': 'us-central1',
            'PROJECT_ID': GCP_PROJECT_ID,
            'BUCKET_NAME': GCP_BUCKET_NAME,
            'GOOGLE_APPLICATION_CREDENTIALS': GOOGLE_APPLICATION_CREDENTIALS
        }
        dataset_name = competition_meta[NAME]
        dataset_name = dataset_name[:13] # only half of name string can be devoted to this to ensure no duplicates
        TIMESTAMP_STR =  datetime.utcnow().strftime('%Y_%m_%d-%H_%M_%S')
        dataset_name = dataset_name + TIMESTAMP_STR
        if SUBSAMPLE in profile:
            suffix = "SAMP" + str(profile[SUBSAMPLE])
            dataset_name = dataset_name[:(32-len(suffix))]
            dataset_name = dataset_name + suffix
        dataset_name = dataset_name[:32]
        print('Fitting GCP Tables...')
        runtime_sec = None
        if 'runtime_hr' in profile:
            print("with time-limit = %s hr..." % profile['runtime_hr'])
            runtime_sec = int(profile['runtime_hr'] * 3600)
        
        (num_models_trained, num_models_ensemble, fit_time, y_preds, y_prob, 
         predict_time, class_order) = gcptables_fit_predict(train_data=train_data, test_data=test_data, 
            label_column=competition_meta[LABEL_COLUMN], problem_type=competition_meta[PROBLEM_TYPE], 
            eval_metric=competition_meta[EVAL_METRIC], output_directory=tmp_dir, runtime_sec=runtime_sec,
            dataset_name=dataset_name, gcp_info=gcp_info
        )
        if competition_meta[EVAL_METRIC] in PREDICT_PROBA_METRICS:
            y_preds = y_prob
        
        metrics['num_models_trained'] = num_models_trained
        metrics['num_models_ensemble'] = num_models_ensemble
        metrics['fit_time'] = fit_time
        metrics['pred_time'] = predict_time
        return y_preds, class_order


# Previously used Predictor class:

class PredictorOLD:

    def __init__(self):
        self.predictor = None

    def fit(self, train_data: task.Dataset, competition_meta: Mapping[str, str], profile: Mapping[str, any], output_directory) -> any:
        # Model must be recorded to {tmp_dir}/models/ to be uploaded to s3
        raise NotImplementedError()

    def get_fit_summary(self) -> Mapping[str, any]:
        raise NotImplementedError()

    def get_leaderboard(self) -> DataFrame:
        # Return dataframe must be in format similar to AutoGluon for easy aggregation:
        # model | score_val | fit_time | pred_time_val | stack_level
        raise NotImplementedError()

    def predict_proba(self, dataset: DataFrame) -> np.array:
        raise NotImplementedError()

    def get_class_labels(self) -> List:
        raise NotImplementedError()


class GluonAutoMLPredictor(PredictorOLD):
    def fit(self, train_data: task.Dataset, competition_meta: Mapping[str, str], profile: Mapping[str, any], output_directory) -> any:
        self.predictor: TabularPredictor = task.fit(
            train_data=train_data,
            label=competition_meta[LABEL_COLUMN],
            eval_metric=competition_meta[EVAL_METRIC],
            problem_type=competition_meta[PROBLEM_TYPE],
            output_directory=output_directory,
            **profile
        )

    def get_fit_summary(self) -> Mapping[str, any]:
        return self.predictor.fit_summary()

    def get_leaderboard(self) -> DataFrame:
        return self.predictor.leaderboard(silent=True)

    def predict_proba(self, dataset: DataFrame) -> np.array:
        return self.predictor.predict_proba(dataset)

    def get_class_labels(self) -> List:
        return self.predictor.class_labels

# Helper functions:
def _convert_dict_values_to_str(d):
    for k, v in d.items():
        if isinstance(v, dict):
            _convert_dict_values_to_str(v)
        else:
            if type(v) not in [list, tuple, int, float, bool, str]:
                v = str(v)
            d.update({k: v})
    return d
