""" Example script to run Auto-WEKA. Must be run from directory containing baselines/ folder. """

import pandas as pd
import numpy as np

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon import TabularPrediction as task

from baselines.autoweka.methods_autoweka import autoweka_fit_predict


# Set arguments:
train_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv'
test_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv'
predict_proba = False
pred_class_and_proba = True
runtime_sec = 60
num_cores = -1
random_state = 0
# Folder containing lib/autoweka/autoweka.jar installed during setup.sh:
autoweka_path = "autoweka_openml_bench/"
# On EC2: autoweka_path = "/home/ubuntu/"

# Specify prediction problem:
label_column = 'class'  # specifies which column do we want to predict 
problem_type = BINARY
eval_metric = 'roc_auc'
output_directory = "autowekatest/" # On EC2: autoweka_path = "/home/ubuntu/autowekaTest/"
# where to save trained models (NOTE: change this every run! should end in slash)


# Load data:
train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500) # subsample for faster demo
print(train_data.head())
test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame

# Run Auto-WEKA:
(num_models_trained, num_models_ensemble, fit_time, 
    y_pred, y_prob, predict_time, class_order) = autoweka_fit_predict(train_data=train_data, test_data=test_data, 
        label_column=label_column, problem_type=problem_type, output_directory=output_directory, autoweka_path=autoweka_path,
        eval_metric=eval_metric, runtime_sec=runtime_sec, random_state=random_state, num_cores=num_cores)


# Can use autogluon.tabular.Predictor to evaluate predictions (assuming metric correctly specified):
ag_predictor = task.fit(task.Dataset(df=train_data), label=label_column, 
        problem_type=problem_type, eval_metric=eval_metric, hyperparameters={'GBM': {'num_boost_round': 2}})
if eval_metric == 'roc_auc':
    preds_toevaluate = y_prob[:,1]
elif eval_metric == 'log_loss':
    preds_toevaluate = y_prob
else:
    preds_toevaluate = y_pred

perf = ag_predictor.evaluate_predictions(test_data[label_column], preds_toevaluate) # use y_prob or y_prob[:,1] instead of y_pred for metrics like log_loss or roc_auc

print("Auto-WEKA test performance: %s" % perf)
print("Number of models trained during Auto-WEKA fit(): %s" % num_models_trained)
print("Auto-WEKA ensemble-size used at inference-time: %s" % num_models_ensemble)
print("Auto-WEKA fit runtime: %s" % fit_time)
print("Auto-WEKA predict runtime: %s" % predict_time)

## To repeat for multi-class task, change above:
label_column = 'occupation'
problem_type = MULTICLASS # Note the evaluation code after y_pred will have bugs because some test-examples are missing labels.
eval_metric = 'log_loss'
output_directory = "autoweka/multiclass_run2/"

train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(5000) # subsample for faster demo
print(train_data.head())
test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame
# Only keep top 4 classes:
class_cnts = train_data[label_column].value_counts()
allowed_labels = list(class_cnts.keys())[:4]
train_data = train_data[train_data[label_column].isin(allowed_labels)]
test_data = test_data[test_data[label_column].isin(allowed_labels)]


## To repeat for regression task, change above:
label_column = 'age'
problem_type = REGRESSION
eval_metric = 'r2'
output_directory = "autoweka/regression_run/"

train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500) # subsample for faster demo
print(train_data.head())
test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame






