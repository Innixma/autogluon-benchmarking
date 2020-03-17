""" Example script to run auto-sklearn. 
    Note: auto-sklearn can only be run on Linux, not Mac.
"""

import pandas as pd
import numpy as np
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

from baselines.autosklearn_base.autosklearn_base import AutoSklearnBaseline

# Set arguments:
output_directory = 'autosklearn_models/' # where to save trained models
train_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv'
test_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv'
predict_proba = False
pred_class_and_proba = True
runtime_sec = 120
num_cores = None

# Specify prediction problem:
label_column = 'class' # specifies which column do we want to predict
problem_type = BINARY
eval_metric = 'roc_auc'

# Load data:
train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500) # subsample for faster demo
print(train_data.head())

test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame
y_test = test_data[label_column] 
test_data.drop([label_column], axis=1, inplace=True) # If you do not remove test-data labels, then predictAutoSklearn() may return less predictions than datapoints (preprocessing filters out rowws with badly-formatted labels) 

# Run auto-sklearn:
autosk = AutoSklearnBaseline()

num_models_trained, num_models_ensemble, fit_time = autosk.fit(train_data=train_data, label_column=label_column,
        problem_type=problem_type, eval_metric=eval_metric, runtime_sec=runtime_sec, num_cores=num_cores)


y_pred, y_prob, predict_time = autosk.predict(test_data, 
        predict_proba=predict_proba, pred_class_and_proba=pred_class_and_proba)
class_order = autosk.classes # ordering of classes corresponding to columns of y_prob

# Can use autogluon.tabular.Predictor to evaluate predictions (assuming metric correctly specified):
perf = autosk.ag_predictor.evaluate_predictions(y_test, y_pred) # use y_prob or y_prob[:,1] for metrics like log_loss or roc_auc

print("auto-sklearn test performance: %s" % perf)
print("Number of models trained during auto-sklearn fit(): %s" % num_models_trained)
print("auto-sklearn ensemble-size used at inference-time: %s" % num_models_ensemble)
print("auto-sklearn fit runtime: %s" % fit_time)
print("auto-sklearn predict runtime: %s" % predict_time)
if problem_type != REGRESSION:
    print("auto-sklearn class order: " + str(class_order))


## To repeat for regression task, change above:
label_column = 'age'
problem_type = REGRESSION
eval_metric = 'mean_absolute_error'


## To repeat for multi-class task, change above:
label_column = 'occupation'
problem_type = MULTICLASS # Note the evaluation code after y_pred will have bugs because some test-examples are missing labels.
eval_metric = 'accuracy'

