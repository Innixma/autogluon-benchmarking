import time, warnings, os, tempfile, math
import pandas as pd
from psutil import virtual_memory
import shutil

import autosklearn.classification
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

from ..process_data import processData, autogluon_class_order
from ..model_base import AbstractBaseline

class AutoSklearnBaseline(AbstractBaseline):
    """ Methods to run auto-sklearn. 
        auto-sklearn cannot handle non-float features, so need to use AutoGluon data preprocessing first.
        auto-sklearn must be run on Linux, not Mac.
    """
    def __init__(self):
        super()
        self.classes = None
    
    def fit(self, train_data, label_column, problem_type, output_directory = None, eval_metric = None, 
            runtime_sec = 60, random_state = 0, num_cores = None):
        """ Tries to fit with specified number of cores (4 if unspecified), then falls back to 1 core if this failed. 
        """
        try: 
            (num_models_trained, num_models_ensemble, fit_time) = self.fit_with_cores(train_data, label_column, 
                    problem_type, output_directory=output_directory, eval_metric=eval_metric,
                    runtime_sec=runtime_sec, random_state=random_state, num_cores=num_cores)
        except Exception as e:
            print("autosklearn produced exception:")
            print(e)
            print('Re-trying autosklearn with num_cores = 1...')
            shutil.rmtree(output_directory)
            # output_directory = output_directory + 'num_cores_1_attempt/'
            (num_models_trained, num_models_ensemble, fit_time) = self.fit_with_cores(train_data, label_column, 
                    problem_type, output_directory=output_directory, eval_metric=eval_metric,
                    runtime_sec=runtime_sec, random_state=random_state, num_cores=1)
        return (num_models_trained, num_models_ensemble, fit_time)
    
    def fit_with_cores(self, train_data, label_column, problem_type, output_directory = None, eval_metric = None, 
            runtime_sec = 60, random_state = 0, num_cores = None):
        """ Args:
                num_cores : can be None in which case auto-sklearn uses n_jobs = 4 (safe value as long as your machine has sufficient CPUs). 
                Set = -1 to use all cores, but this is an EXPERIMENTAL feature of auto-sklearn and may crash.
        """
        # Set same configurations used as OpenML AutoML Benchmark:
        os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.gettempdir()
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        if num_cores == -1: # autosk.fit() produces bug: "automl=self._automl[0], IndexError: list index out of range"
            warnings.warn("setting num_cores==-1 in auto-sklearn produces bug, running with num_cores=None instead.")
            num_cores = None
        n_jobs = num_cores
        if n_jobs is None:
            n_jobs = 4
        
        DEFAULT_ML_MEMORY_LIMIT = 3072  # 3072 is autosklearn defaults
        DEFAULT_ENSEMBLE_MEMORY_LIMIT = 1024 # 1024 is autosklearn defaults
        mem = virtual_memory()
        total_gb = mem.total >> 30
        total_memory_limit_mb = (total_gb - 2) * 1000 # Leave 2GB free for OS, as done for h2o.
        # when memory is large enough, we should have:
        # memory_limit_mb = (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb
        ml_memory_limit = max(math.ceil(total_memory_limit_mb/n_jobs), 
                              DEFAULT_ML_MEMORY_LIMIT)
        if ml_memory_limit >= total_memory_limit_mb - DEFAULT_ENSEMBLE_MEMORY_LIMIT:
            ml_memory_limit = max(total_memory_limit_mb - DEFAULT_ENSEMBLE_MEMORY_LIMIT,
                                  DEFAULT_ML_MEMORY_LIMIT)
        remaining_memory = total_memory_limit_mb - ml_memory_limit
        if DEFAULT_ENSEMBLE_MEMORY_LIMIT > remaining_memory:
            ensemble_memory_limit = DEFAULT_ENSEMBLE_MEMORY_LIMIT
        else:
            ensemble_memory_limit = max(math.ceil(remaining_memory - ml_memory_limit),
                                        math.ceil(ml_memory_limit / 3.0),  # default proportions
                                        DEFAULT_ENSEMBLE_MEMORY_LIMIT)
        
        X_train, y_train, ag_predictor = processData(data=train_data, label_column=label_column, 
                                                     problem_type=problem_type, eval_metric=eval_metric)
        print("Fitting auto-sklearn....")
        t0 = time.time()
        # Specify auto-sklearn settings:
        autosk_params = {
            'n_jobs': n_jobs, # = 1 or set = -1 to use all available cores. 
            # If num_cores == -1, autosk.fit() can produce bug: "automl=self._automl[0], IndexError: list index out of range". This is a known bug reported here: https://github.com/automl/auto-sklearn/pull/733
            'seed': random_state,
            'time_left_for_this_task': runtime_sec,
            'per_run_time_limit': int(max(min(360, int(runtime_sec*0.99)), 
                                          runtime_sec/5.0)), # run at least 5 trials if overall runtime > 360.
            'ml_memory_limit': ml_memory_limit,
            'ensemble_memory_limit': ensemble_memory_limit,
            # Note: OpenML AutoML benchmark did not set per_run_time_limit because their datasets are smaller, it is crucial for larger datasets. Here it is set as recommended by auto-sklearn authors.
            # 'ensemble_size': 50
        }
        if output_directory is not None:
            autosk_params['tmp_folder'] = output_directory
            autosk_params['delete_tmp_folder_after_terminate'] = False
        
        if problem_type in [BINARY, MULTICLASS]:
            autosk = autosklearn.classification.AutoSklearnClassifier(**autosk_params)
        elif problem_type == REGRESSION:
            autosk = autosklearn.regression.AutoSklearnRegressor(**autosk_params)
        else:
            raise ValueError("Unknown problem type: %s" % problem_type)
        if eval_metric is None: # Pass in metric to fit()
            autosk.fit(X_train, y_train)
        else:
            autosk_metric = self.convert_metric(eval_metric)
            autosk.fit(X_train, y_train, metric=autosk_metric)
        t1 = time.time()
        fit_time = t1 - t0
        self.ag_predictor = ag_predictor
        self.model = autosk
        num_models_trained = self.numTrainedModels()
        num_models_ensemble = len(autosk.get_models_with_weights())
        # ag_leaderboard = ag_predictor.leaderboard(silent=True) # tracks extra time (sec) used preprocessing data
        # sec_wasted = ag_leaderboard.fit_time[0] # we don't need to compute wasted time since we are timing inside fit()
        return (num_models_trained, num_models_ensemble, fit_time)
    
    def predict(self, test_data, predict_proba = False, pred_class_and_proba = False):
        """ Use pred_class_and_proba to produce both predicted probabilities and predicted classes.
            If this is regression problem, predict_proba and pred_class_and_proba are disregarded.
            Label column should not be present in test_data.
            
            Returns: Tuple (y_pred, y_prob, inference_time) where any element may be None.
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
            self.classes = autogluon_class_order(self.ag_predictor) # ordering of classes corresponding to columns of y_prob
        
        t1 = time.time()
        predict_time = t1 - t0
        return (y_pred, y_prob, predict_time)
    
    def convert_metric(self, metric):
        """Converts given metric to appropriate auto-sklearn 'metric' param, used to guide training.
           auto-sklearn scores follow the sklearn.metrics API, c.f. https://automl.github.io/auto-sklearn/master/api.html#api
           An instance of autosklearn.metrics.Scorer as created by autosklearn.metrics.make_scorer(). These are the Built-in Metrics, 
           with options listed here:  https://automl.github.io/auto-sklearn/master/api.html#built-in-metrics
           
           Args:
                metric : str
                    May take one of the following values:
        """
        autosk_metrics = { # dict mapping autogluon metric to the metric used by auto-sklearn
            'accuracy': autosklearn.metrics.accuracy,
            'f1': autosklearn.metrics.f1,
            'log_loss': autosklearn.metrics.log_loss,
            'roc_auc': autosklearn.metrics.roc_auc,
            'balanced_accuracy': autosklearn.metrics.balanced_accuracy,
            'precision': autosklearn.metrics.precision,
            'recall': autosklearn.metrics.recall,
            'mean_squared_error': autosklearn.metrics.mean_squared_error,
            'median_absolute_error': autosklearn.metrics.median_absolute_error,
            'mean_absolute_error': autosklearn.metrics.mean_absolute_error,
            'r2': autosklearn.metrics.r2,
        }
        if metric in autosk_metrics:
            return autosk_metrics[metric]
        else:
            warnings.warn("Unknown metric will not be used by auto-sklearn: %s" % metric)
            return None
    
    def numTrainedModels(self): 
        """ Returns None if number of trained models cannot be found. """
        autosk = self.model
        autosk_summary_str = autosk.sprint_statistics()
        list_str = autosk_summary_str.split("\n")
        identifier_str = "Number of target algorithm runs:" # how we find the right string to extract this value from.
        inds = [i for i, val in enumerate(list_str) if identifier_str in val] 
        if len(inds) != 1:
            warnings.warn("Failed to get number of trained models from AutoSKLearn; sprint_statistics() returned unexpected formatted string")
            return None
        num_runs_str = list_str[inds[0]].strip()
        split_str = num_runs_str.split(identifier_str)
        if len(split_str) != 2:
            warnings.warn("Failed to get number of trained models from AutoSKLearn; sprint_statistics() returned unexpected formatted string")
            return None
        num_runs = int(split_str[1].strip())
        return num_runs

