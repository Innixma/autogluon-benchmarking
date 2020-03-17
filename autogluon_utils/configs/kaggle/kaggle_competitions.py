from typing import Type, Mapping

from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

from autogluon_utils.benchmarking.kaggle.postprocessors.kaggle_postprocessor import StandardPostProcessor, WalmartRecritingPostProcessor, \
    LogLabelTransformPostProcessor
from autogluon_utils.benchmarking.kaggle.preprocessors.kaggle_preprocessor import IeeeFraudDetectionPreProcessor, \
    StandardPreProcessor, \
    WalmartRecruitingPreProcessor, NestedUnzipperPreprocessor, LogLabelTransformPreProcessor, SberbankPreProcessor
from autogluon_utils.benchmarking.kaggle.predictors.predictors import *
from autogluon_utils.configs.kaggle.constants import *
import autogluon_utils.utils.config as config

CONFIG = config.load_config()

KAGGLE_DATASET_FARM = CONFIG['s3_path']

PREDICTORS: Mapping[str, Type[Predictor]] = {
    # Use url-friendly names
    'autogluon_predictor': AutoGluonPredictor,
    'h2o_predictor': h2oPredictor,
    'autosklearn_predictor': autosklearnPredictor,
    'autoweka_predictor': autowekaPredictor,
    'tpot_predictor': tpotPredictor,
    'gcp_predictor': gcpPredictor,
    # OLD predictor: 'autogluon_tabular': GluonAutoMLPredictor,
}

NO_TIME_LIMITS = None
# TODO: consider adding another key "POSITIVE_CLASS" for binary problems in case it does not match target column
KAGGLE_COMPETITIONS = [
    {
        NAME: 'ieee-fraud-detection',
        LABEL_COLUMN: 'isFraud',
        EVAL_METRIC: 'roc_auc',
        PROBLEM_TYPE: BINARY,
        INDEX_COL: 'TransactionID',
        PRE_PROCESSOR: IeeeFraudDetectionPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'porto-seguro-safe-driver-prediction',
        LABEL_COLUMN: 'target',
        EVAL_METRIC: 'roc_auc',
        # Competition metric is normalized gini index, we can instead use roc_auc, since gini = 2*roc_auc - 1
        # Higher is still better for normalized gini as well as roc_auc
        PROBLEM_TYPE: BINARY,
        INDEX_COL: 'id',
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'walmart-recruiting-trip-type-classification',
        LABEL_COLUMN: 'TripType',
        EVAL_METRIC: 'log_loss',
        PROBLEM_TYPE: MULTICLASS, # Note: original class names are referred to as TripType_class in submission file.
        INDEX_COL: None,
        PRE_PROCESSOR: WalmartRecruitingPreProcessor(),
        POST_PROCESSOR: WalmartRecritingPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'allstate-claims-severity',
        LABEL_COLUMN: 'loss',
        EVAL_METRIC: 'mean_absolute_error',
        PROBLEM_TYPE: REGRESSION,
        INDEX_COL: 'id',
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'santander-customer-transaction-prediction',
        LABEL_COLUMN: 'target',
        EVAL_METRIC: 'roc_auc',
        PROBLEM_TYPE: BINARY,
        INDEX_COL: 'ID_code',
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'santander-customer-satisfaction',
        LABEL_COLUMN: 'TARGET',
        EVAL_METRIC: 'roc_auc',
        PROBLEM_TYPE: BINARY,
        INDEX_COL: 'ID',
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'santander-value-prediction-challenge',
        LABEL_COLUMN: 'target',
        EVAL_METRIC: 'mean_squared_error',
        # The leaderboard uses root mean squared logarithmic error. To labels, we perform log-transform and inverse in pre/post-processing, so predictors can just target MSE.
        PROBLEM_TYPE: REGRESSION,
        INDEX_COL: 'ID',
        PRE_PROCESSOR: LogLabelTransformPreProcessor(label_column='target'),
        POST_PROCESSOR: LogLabelTransformPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'microsoft-malware-prediction',
        LABEL_COLUMN: 'HasDetections',
        EVAL_METRIC: 'roc_auc',
        PROBLEM_TYPE: BINARY,
        INDEX_COL: 'MachineIdentifier',
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'mercedes-benz-greener-manufacturing',
        LABEL_COLUMN: 'y',
        EVAL_METRIC: 'r2',
        PROBLEM_TYPE: REGRESSION,
        INDEX_COL: 'ID',
        FETCH_PROCESSOR: NestedUnzipperPreprocessor(),
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'house-prices-advanced-regression-techniques',
        LABEL_COLUMN: 'SalePrice',
        EVAL_METRIC: 'mean_squared_error',
        # The leaderboard uses root mean squared logarithmic error. To labels, we perform log-transform and inverse in pre/post-processing, so predictors can just target MSE.
        PROBLEM_TYPE: REGRESSION,
        INDEX_COL: 'Id',
        PRE_PROCESSOR: LogLabelTransformPreProcessor(label_column='SalePrice'),
        POST_PROCESSOR: LogLabelTransformPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'otto-group-product-classification-challenge',
        LABEL_COLUMN: 'target',
        EVAL_METRIC: 'log_loss',
        PROBLEM_TYPE: MULTICLASS,
        INDEX_COL: 'id',
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
    {
        NAME: 'bnp-paribas-cardif-claims-management',
        LABEL_COLUMN: 'target',
        LABEL_COLUMN_PRED: 'PredictedProb',
        EVAL_METRIC: 'log_loss',
        PROBLEM_TYPE: BINARY,
        INDEX_COL: 'ID',
        FETCH_PROCESSOR: NestedUnzipperPreprocessor(),
        PRE_PROCESSOR: StandardPreProcessor(),
        POST_PROCESSOR: StandardPostProcessor(),
        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    },
#    {
#        NAME: 'sberbank-russian-housing-market',
#        LABEL_COLUMN: 'price_doc',
#        EVAL_METRIC: 'mean_squared_error',
#        # The leaderboard uses root mean squared logarithmic error. To labels, we perform log-transform and inverse in pre/post-processing, so predictors can just target MSE.
#        PROBLEM_TYPE: REGRESSION,
#        INDEX_COL: 'id',
#        FETCH_PROCESSOR: NestedUnzipperPreprocessor(),
#        PRE_PROCESSOR: SberbankPreProcessor(),
#        POST_PROCESSOR: StandardPostProcessor(),
#        FITTING_TIME_LIMITS: NO_TIME_LIMITS,
#    },
]


def eval_metric_lower_is_better(competition: str):
    competitions = {x[NAME]: x for x in KAGGLE_COMPETITIONS}
    assert competition in competitions
    eval_metric = competitions[competition][EVAL_METRIC]
    if eval_metric == 'log_loss':
        return True
    elif eval_metric == 'root_mean_squared_error':
        return True
    elif eval_metric == 'r2':
        return False
    elif eval_metric == 'roc_auc':
        return False
    elif eval_metric == 'mean_absolute_error':
        return True
    elif eval_metric == 'loss':
        return True
    elif eval_metric == 'mean_squared_error':
        return True
    assert False, 'uknown metric'

# These profiles contain the config options for all autoML methods.
FITTING_PROFILES = {
    PROFILE_FAST: { # Use this profile for quick test.
        'runtime_hr': 0.015,
        SUBSAMPLE: 200,
        # AutoGluon params:
        'verbosity': 3,
        'hyperparameter_tune': False,
        'auto_stack': False,
    },
    PROFILE_FAST2: { # Use this profile for GCP test.
        'runtime_hr': 0.05,
        SUBSAMPLE: 1001,
        # AutoGluon params:
        'verbosity': 3,
        'hyperparameter_tune': False,
        'auto_stack': False,
    },
    PROFILE_FULL_1HR: { # Use this profile for official 1hr run
        'runtime_hr': 1,
        # AutoGluon params:
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    PROFILE_FULL_4HR: { # Use this profile for official 4hr run
        'runtime_hr': 4,
        # AutoGluon params:
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    PROFILE_FULL_8HR: { # Use this profile for official 8hr run
        'runtime_hr': 8,
        # AutoGluon params:
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    PROFILE_FULL_12HR: { # Use this profile for official 12hr run
        'runtime_hr': 12,
        # AutoGLuon params:
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    PROFILE_FULL_24HR: { # Use this profile for official 24hr run
        'runtime_hr': 24,
        # AutoGLuon params:
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    PROFILE_FULL_32HR: { # Use this profile for official 32hr run
        'runtime_hr': 32,
        # AutoGLuon params:
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    # Old profiles for autogluon runs:
    PROFILE_FULL: {
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': True,
    },
    PROFILE_LEVEL_1_ONLY: {
        'hyperparameter_tune': False,
        'verbosity': 3,
        'auto_stack': False,
    },
    PROFILE_FULL_NO_STACKING_WITH_HPO: {
        'hyperparameter_tune': True,
        'verbosity': 3,
    },
    PROFILE_FULL_WITH_HPO: {
        'hyperparameter_tune': True,
        'verbosity': 3,
        'auto_stack': True,
    },
}
