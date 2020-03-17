""" Example script to run GCP AutoML Tables. Must be run from directory containing baselines/ folder. """

import pandas as pd
import numpy as np

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon import TabularPrediction as task

from baselines.gcp.methods_gcp import *

# Set arguments:
gcp_info = { # GCP configuration info. You should only change: GOOGLE_APPLICATION_CREDENTIALS to location of GCP key file.
    'COMPUTE_REGION': "us-central1",
    'PROJECT_ID': "automl-264518", # TODO: Put your project ID here.
    'BUCKET_NAME': "tabulardata", # TODO: Put your GCS bucket name here.
    'GOOGLE_APPLICATION_CREDENTIALS': '<Your_GCP_key>.json', # TODO: Put your GCP key file-path here.
}

runtime_sec = 3600
fit_model = True
make_predictions = True

# Specify prediction problem:
problem_type = BINARY
label_column = "class"
eval_metric = 'log_loss' # None
dataset_name = "income_binary" # should be unique for every GCP run
output_directory = "gcptest/"+dataset_name


# Load data:
train_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv'
test_file = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv'
train_data = load_pd.load(train_file) # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(1001)
test_data = load_pd.load(test_file) # can be local CSV file as well, returns Pandas DataFrame
y_test = test_data[label_column] 
test_data.drop([label_column], axis=1, inplace=True)


# Run GCP AutoML Tables:
num_models_trained, num_models_ensemble, fit_time, y_pred, y_prob, predict_time, class_order = gcptables_fit_predict(
        train_data=train_data, test_data=test_data, dataset_name=dataset_name, label_column=label_column, 
        problem_type=problem_type, eval_metric=eval_metric, output_directory=output_directory, gcp_info=gcp_info,
        runtime_sec=runtime_sec, fit_model=fit_model, model_name=None, make_predictions=make_predictions
    )


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

print("GCP test performance: %s" % perf)
print("GCP fit runtime: %s" % fit_time)
print("GCP predict runtime: %s" % predict_time)


## To repeat for regression task, change above:
label_column = 'age'
problem_type = REGRESSION
eval_metric = 'r2'
dataset_name = "age_regress"
output_directory = "gcptest/"+dataset_name


## To repeat for multi-class task, change above:
label_column = 'occupation'
problem_type = MULTICLASS # Note the evaluation code after y_pred will have bugs because some test-examples are missing labels.
eval_metric = 'log_loss'
dataset_name = "occupation_multi"
output_directory = "gcptest/"+dataset_name

