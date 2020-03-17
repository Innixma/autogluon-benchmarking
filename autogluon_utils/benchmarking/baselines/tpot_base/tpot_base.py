import time, warnings
import psutil
import tpot
from tpot import TPOTClassifier, TPOTRegressor
import pandas as pd
import numpy as np
from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

from ..process_data import processData, autogluon_class_order
from ..model_base import AbstractBaseline

class TPOTBaseline(AbstractBaseline):
    """ Methods to run TPOT. 
        TPOT cannot handle non-float features, so need to use AutoGluon data preprocessing first.
    """
    def __init__(self):
        super()
        self.classes = None
    
    def fit(self, train_data, label_column, output_directory, problem_type, 
            eval_metric = None, runtime_sec = 60, random_state = 0, num_cores = -1):
        X_train, y_train, ag_predictor = processData(data=train_data, label_column=label_column, 
                                                     problem_type=problem_type, eval_metric=eval_metric)
        t0 = time.time()
        # TPOT settings:
        tpot_params = {
            'n_jobs': num_cores, # or set = -1 to use all available cores.
            'random_state': random_state,
            'max_time_mins': runtime_sec/60.0,
            'max_eval_time_mins': 5 # Not specified in the OpenML AutoML benchmark code, but TPOT authors recommended specifying this value.
            # These can all be left at default values:
            # 'verbosity': 2,
            # 'generations': 100,
            # 'population_size': 100,
            # 'max_eval_time_mins': None,
        }
        # For large run-times, consider resetting max_eval_time_mins as suggested by TPOT authors:
        N_JOBS = tpot_params['n_jobs']
        if N_JOBS == -1:
            N_JOBS = psutil.cpu_count(logical=False)
        if tpot_params['max_time_mins'] * N_JOBS > 1500: # note that by default: population_size = 100, offspring_size = None
            tpot_params['max_eval_time_mins'] = tpot_params['max_time_mins']*N_JOBS/(3.0*100.0)
        if eval_metric is not None: # Pass in metric to fit()
            tpot_metric = self.convert_metric(eval_metric)
            if tpot_metric is not None:
                tpot_params['scoring'] = tpot_metric
        
        if problem_type in [BINARY, MULTICLASS]:
            tpot = TPOTClassifier(**tpot_params)
        elif problem_type == REGRESSION:
            tpot = TPOTRegressor(**tpot_params)
        else:
            raise ValueError("Unknown problem type: %s" % problem_type)
        tpot.fit(X_train, y_train)
        t1 = time.time()
        fit_time = t1 - t0
        num_models_trained =len(tpot.evaluated_individuals_)
        try:
            num_models_ensemble = len(tpot.fitted_pipeline_)  # does not work if runtime is too short.
        except Exception:
            num_models_ensemble = -1
        # ag_leaderboard = ag_predictor.leaderboard(silent=True) # tracks extra time (sec) used preprocessing data
        # ag_sec_wasted = ag_leaderboard.fit_time[0]
        self.model = tpot
        self.ag_predictor = ag_predictor # needed for preprocessing future data
        return (num_models_trained, num_models_ensemble, fit_time)
    
    def predict(self, test_data, predict_proba = False, pred_class_and_proba = False):
        """ Use pred_class_and_proba to produce both predicted probabilities and predicted classes.
            If this is regression problem, predict_proba and pred_class_and_proba are disregarded.
            Label column should not be present in test_data.
            
            Returns: Tuple (y_pred, y_proba, inference_time) where any element may be None.
            y_prob is a 2D numpy array of predicted probabilities, where each column represents a class. The ith column represents the class found via: self.classes[i].
        """
        X_test, y_test, _ = processData(data=test_data, label_column=self.ag_predictor._learner.label, ag_predictor=self.ag_predictor)
        if self.ag_predictor.problem_type == REGRESSION:
            pred_class_and_proba = False
            predict_proba = False
        y_pred = None
        y_prob = None
        t0 = time.time()
        if (not predict_proba) or pred_class_and_proba:
            y_pred = self.model.predict(X_test)
            y_pred = self.ag_predictor._learner.label_cleaner.inverse_transform(pd.Series(y_pred))
        if predict_proba or pred_class_and_proba:
            y_prob = self.model.predict_proba(X_test)
            y_prob = self.ag_predictor._learner.label_cleaner.inverse_transform_proba(y_prob) # handles rare classes possibly omitted during processing
            self.classes = autogluon_class_order(self.ag_predictor)  # ordering of classes corresponding to columns of y_prob
        t1 = time.time()
        predict_time = t1 - t0
        return (y_pred, y_prob, predict_time)
    
    def convert_metric(self, metric):
        """Converts given metric to appropriate TPOT 'scoring' param, used to guide training.
           TPOT scores follow the sklearn.metrics API, c.f. http://epistasislab.github.io/tpot/api/#classification.
           
           Args:
                metric : str
                    For classification: can be one of: 'acc', 'auc', 'f1', 'logloss', 'balanced_acc'
                    For regression, can be one of: 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error'
        """
        tpot_metrics = { # dict mapping metric_str to the str used by TPOT:
            'accuracy': 'accuracy',
            'f1': 'f1',
            'log_loss': 'neg_log_loss',
            'roc_auc': 'roc_auc',
            'balanced_accuracy': 'balanced_accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'mean_squared_error': 'neg_mean_squared_error',
            'median_absolute_error': 'neg_median_absolute_error',
            'mean_absolute_error': 'neg_mean_absolute_error',
            'r2': 'r2',
        }
        if metric in tpot_metrics:
            return tpot_metrics[metric]
        else:
            warnings.warn("Unknown metric will not be used by TPOT: %s" % metric)
            return None
