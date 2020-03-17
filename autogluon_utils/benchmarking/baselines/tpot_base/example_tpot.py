""" Example script to run TPOT. 
    Must be run from directory containing baselines/ folder.
"""

import pandas as pd
import numpy as np
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

from baselines.tpot_base.tpot_base import TPOTBaseline

# Set arguments:
output_directory = 'tpot_models/' # where to save trained models
train_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv'
test_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv'
predict_proba = False
pred_class_and_proba = True
runtime_sec = 120
num_cores = -1

# Specify prediction problem:
label_column = 'class' # specifies which column do we want to predict
problem_type = BINARY
eval_metric = 'roc_auc' # set to None to avoid using metric

# Load data:
train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(100) # subsample for faster demo
print(train_data.head())

test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame
y_test = test_data[label_column]
test_data.drop([label_column], axis=1, inplace=True) # If you do not remove test-data labels, then predictTPOT() may return less predictions than datapoints (preprocessing filters out rowws with badly-formatted labels) 

# Run TPOT: 
tpot_base = TPOTBaseline()

num_models_trained, num_models_ensemble, fit_time = tpot_base.fit(train_data=train_data, label_column=label_column,
        output_directory=output_directory, problem_type=problem_type, eval_metric=eval_metric,
        runtime_sec=runtime_sec, num_cores=num_cores)

y_pred, y_prob, predict_time = tpot_base.predict(test_data, 
        predict_proba=predict_proba, pred_class_and_proba=pred_class_and_proba)
class_order = tpot_base.classes # ordering of classes corresponding to columns of y_prob

# Can use autogluon.tabular.Predictor to evaluate predictions (assuming metric correctly specified):
if eval_metric == 'roc_auc':
    preds_toevaluate = y_prob[:,1]
elif eval_metric == 'log_loss':
    preds_toevaluate = y_prob
else:
    preds_toevaluate = y_pred

perf = tpot_base.ag_predictor.evaluate_predictions(y_test, preds_toevaluate) # use y_prob or y_prob[:,1] instead of y_pred for metrics like log_loss or roc_auc

print("TPOT test performance: %s" % perf)
print("Number of models trained during TPOT fit(): %s" % num_models_trained)
print("TPOT ensemble-size used at inference-time: %s" % num_models_ensemble)
print("TPOT fit runtime: %s" % fit_time)
print("TPOT predict runtime: %s" % predict_time)
if problem_type != REGRESSION:
    print("TPOT class order: " + str(class_order))

## To repeat for regression task, change above:
label_column = 'age'
problem_type = REGRESSION
eval_metric = 'mean_squared_error'


## To repeat for multi-class task, change above:
label_column = 'occupation'
problem_type = MULTICLASS # Note the evaluation code after y_pred will have bugs because some test-examples are missing labels.
eval_metric = 'balanced_accuracy'

