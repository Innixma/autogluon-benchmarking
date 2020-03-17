""" Example script to run h2o """

import pandas as pd
import numpy as np
import h2o
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon import TabularPrediction as task

from baselines.h2o_base.h2o_base import H2OBaseline

# Set arguments:
output_directory = 'H2O_models/' # where to save trained models
train_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv'
test_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv'
predict_proba = False
pred_class_and_proba = True
runtime_sec = 60
num_cores = -1

# Specify prediction problem:
label_column = 'class' # specifies which column do we want to predict
problem_type = BINARY
eval_metric = 'log_loss'

# Load data:
train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500) # subsample for faster demo
print(train_data.head())

test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame
y_test = test_data[label_column] 
test_data.drop([label_column], axis=1, inplace=True)

# Run h2o:
h2o_model = H2OBaseline()

num_models_trained, num_models_ensemble, fit_time = h2o_model.fit(train_data=train_data,
                    label_column=label_column, problem_type=problem_type, eval_metric=eval_metric, 
                    runtime_sec=runtime_sec, num_cores=num_cores)

y_pred, y_prob, predict_time = h2o_model.predict(test_data, 
        predict_proba=predict_proba, pred_class_and_proba=pred_class_and_proba)

class_order = h2o_model.classes # ordering of classses corresponding to columns of y_prob

# Can use autogluon.tabular.Predictor to evaluate predictions (assuming metric correctly specified):
ag_predictor = task.fit(task.Dataset(df=train_data), label=label_column, 
        problem_type=problem_type, eval_metric=eval_metric, hyperparameters={'GBM': {'num_boost_round': 2}})
if eval_metric == 'roc_auc':
    preds_toevaluate = y_prob[:,1]
elif eval_metric == 'log_loss':
    preds_toevaluate = y_prob
else:
    preds_toevaluate = y_pred

perf = ag_predictor.evaluate_predictions(y_test, preds_toevaluate) # use y_prob or y_prob[:,1] instead of y_pred for metrics like log_loss or roc_auc

print("H2O test performance: %s" % perf)
print("Number of models trained during H2O fit(): %s" % num_models_trained)
print("H2O ensemble-size used at inference-time: %s" % num_models_ensemble)
print("H2O fit runtime: %s" % fit_time)
print("H2O predict runtime: %s" % predict_time)


## To repeat for regression task, change above:
label_column = 'age'
problem_type = REGRESSION
eval_metric = 'r2'


## To repeat for multi-class task, change above:
label_column = 'occupation'
problem_type = MULTICLASS # Note the evaluation code after y_pred will have bugs because some test-examples are missing labels.
eval_metric = 'log_loss'
