import time, warnings
import pandas as pd
import numpy as np
from psutil import virtual_memory
import h2o
from h2o.automl import H2OAutoML
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

from ..model_base import AbstractBaseline

class H2OBaseline(AbstractBaseline):
    """ Methods to run h2o AutoML. """
    def __init__(self):
        super()
        self.class_prefix = 'cls_'
        self.class_suffix = '_cls'
        self.label_column = None
        self.label_type = None
        self.problem_type = None
        self.classes = None

    def fit(self, train_data, label_column, problem_type, output_directory = None,
            eval_metric = None, runtime_sec = 60, random_state = 0, num_cores = 1):
        self.label_column = label_column
        self.label_type = train_data[label_column].dtype
        self.problem_type = problem_type

        if self.problem_type in [BINARY, MULTICLASS]:
            self.add_label_prefix(train_data)

        # Set h2o memory limits recommended by the authors (recommended leaving 2 GB free for the operating system):
        mem = virtual_memory()
        total_gb = mem.total >> 30
        memory_limit_mb = (total_gb - 2) * 1000
        t0 = time.time()
        h2o.init(nthreads=num_cores, log_dir=output_directory, 
            min_mem_size=str(memory_limit_mb)+"M", max_mem_size=str(memory_limit_mb)+"M",
                )
        train = h2o.H2OFrame(train_data)
        if problem_type in [BINARY, MULTICLASS]:
            train[label_column] = train[label_column].asfactor() # ensure h2o knows this is not regression
        x = list(train.columns)
        if label_column not in x:
            raise ValueError("label_column must be present in training data: %s" % label_column)
        x.remove(label_column)
        # H2O settings:
        training_params = {
            'max_runtime_secs': runtime_sec,
            'seed': random_state,
            # TODO
        }
        if eval_metric is not None: # Pass in metric to fit()
            h2o_metric = self.convert_metric(eval_metric)
            if h2o_metric is not None:
                training_params['sort_metric'] = h2o_metric
                if eval_metric != 'roc_auc': # h2o authors do not recommend using AUC for early-stopping in binary classification and suggest using default instead. We also empirically verified this works better as well.
                    training_params['stopping_metric'] = h2o_metric  # TODO: Not used in AutoMLBenchmark! Do we keep this? Authors mentioned to do this in email.
            else:
                warnings.warn("Specified metric is unknown to h2o. Fitting h2o without supplied evaluation metric instead.")
        h2o_model = H2OAutoML(**training_params)
        h2o_model.train(x=x, y=label_column, training_frame=train)
        if self.problem_type in [BINARY, MULTICLASS]:
            self.remove_label_prefix(train_data)
        t1 = time.time()
        fit_time = t1 - t0
        num_models_trained = len(h2o_model.leaderboard)
        # Get num_models_ensemble:
        if not h2o_model.leader:
            raise AssertionError("H2O could not produce any model in the requested time.")

        best_model = h2o_model.leader
        if 'StackedEnsemble' not in best_model._id:
            num_models_ensemble = 1
        else:
            model_ids = list(h2o_model.leaderboard['model_id'].as_data_frame().iloc[:,0])
            # Get the Stacked Ensemble model if it is the best:
            se = h2o.get_model(best_model._id)
            # Get the Stacked Ensemble metalearner GLM model:
            metalearner = h2o.get_model(se.metalearner()['name'])
            num_models_ensemble = int(metalearner._model_json['output']['model_summary']['number_of_active_predictors'][0])
        self.model = h2o_model
        return (num_models_trained, num_models_ensemble, fit_time)
    
    def predict(self, test_data, predict_proba = False, pred_class_and_proba = False):
        """ Use pred_class_and_proba to produce both predicted probabilities and predicted classes.
            If this is regression problem, predict_proba and pred_class_and_proba are disregarded.
            Label column should not be present in test_data.
            
            Returns: Tuple (y_pred, y_prob, inference_time) where any element may be None.
            y_prob is a 2D numpy array of predicted probabilities, where each column represents a class. The ith column represents the class found via: self.classes[i]
        """
        h2o_model = self.model
        if self.problem_type == REGRESSION:
            pred_class_and_proba = False
            predict_proba = False
        y_pred = None
        y_prob = None
        t0 = time.time()
        test = h2o.H2OFrame(test_data)
        preds_df = h2o_model.predict(test).as_data_frame(use_pandas=True)
        t1 = time.time()
        predict_time = t1 - t0
        if self.problem_type is not REGRESSION:
            self.classes = preds_df.columns.tolist()[1:]
            if self.problem_type in [BINARY, MULTICLASS]:
                self.classes = self.remove_label_prefix_class(self.classes)

        if (not predict_proba) or pred_class_and_proba:
            y_pred = preds_df.iloc[:, 0]
            # print(y_pred[:5])
            if self.problem_type in [BINARY, MULTICLASS]:
                y_pred = pd.Series(self.remove_label_prefix_class(list(y_pred.values)), index=y_pred.index)
            # print(y_pred[:5])

        if predict_proba or pred_class_and_proba:
            y_prob = preds_df.iloc[:, 1:].values
        
        # Shutdown H2O before returning value:
        if h2o.connection():
            h2o.remove_all()
            h2o.connection().close()
        if h2o.connection().local_server:
            h2o.connection().local_server.shutdown()
        return (y_pred, y_prob, predict_time)
    
    def convert_metric(self, metric):
        """Converts given metric to appropriate h2o metric used for sort_metric.
           Args:
                metric : str
                    May take one of the following values: 
        """
        metrics_map = {  # Mapping of benchmark metrics to H2O metrics
            'accuracy': 'AUTO',
            'f1': 'auc',
            'log_loss': 'logloss',
            'roc_auc': 'auc',
            'balanced_accuracy': 'mean_per_class_error',
            'precision': 'auc',
            'recall': 'auc',
            'mean_squared_error': 'mse',
            'root_mean_squared_error': 'mse',
            'median_absolute_error': 'mae',
            'mean_absolute_error': 'mae',
            'r2': 'deviance',
        }
        if metric in metrics_map:
            return metrics_map[metric]
        else:
            warnings.warn("Unknown metric will not be used by h2o: %s" % metric)
            return None

    # Present to deal with defect in H2O regarding altering of class label names
    def add_label_prefix(self, df):
        # print(df[self.label_column].iloc[0])
        df[self.label_column] = [self.class_prefix + str(label[0]) + self.class_suffix for label in zip(df[self.label_column])]
        # print(df[self.label_column].iloc[0])

    # Present to deal with defect in H2O regarding altering of class label names
    def remove_label_prefix(self, df):
        length_to_remove_prefix = len(self.class_prefix)
        length_to_remove_suffix = len(self.class_suffix)
        # print(df[self.label_column].iloc[0])
        df[self.label_column] = [label[0][length_to_remove_prefix:] for label in zip(df[self.label_column])]
        df[self.label_column] = [label[0][:-length_to_remove_suffix] for label in zip(df[self.label_column])]
        # print(df[self.label_column].iloc[0])
        df[self.label_column] = df[self.label_column].astype(self.label_type)

    # Present to deal with defect in H2O regarding altering of class label names
    def remove_label_prefix_class(self, class_name_list):
        length_to_remove_prefix = len(self.class_prefix)
        length_to_remove_suffix = len(self.class_suffix)
        # print(class_name_list)
        class_name_list = [label[length_to_remove_prefix:] for label in class_name_list]
        class_name_list = [label[:-length_to_remove_suffix] for label in class_name_list]
        # print(class_name_list)
        class_name_list = np.array(class_name_list, dtype=self.label_type)
        class_name_list = class_name_list.tolist()
        # print(class_name_list)
        return class_name_list
