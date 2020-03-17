TASK = 'task'
FRAMEWORK = 'framework'
RESULT = 'result'
DURATION = 'duration'
PREDICT_DURATION = 'predict_duration'

TIME_TRAIN_S = 'time_train_s'
TIME_INFER_S = 'time_infer_s'

DATASET = 'dataset'
FOLD = 'fold'
RANK = 'rank'

METRIC_ERROR = 'metric_error'
METRIC_SCORE = 'metric_score'
PROBLEM_TYPE = 'problem_type'

NUM_CLASSES = 'num_classes'
NUM_COLS = 'num_cols'
NUM_ROWS = 'num_rows'
DATETIME = 'datetime'

ERROR_COUNT = 'error_count'
RANK_1 = 'rank=1_count'

BESTDIFF = 'bestdiff'

LOSS_RESCALED = 'loss_rescaled'


# Official automl system names for paper:
SYSTEM_NAMES = {
    'autogluon_1h': 'AutoGluon (1h)',
    'GCPTables_1h': 'GCP-Tables (1h)',
    'autosklearn_1h': 'auto-sklearn (1h)',
    'H2OAutoML_1h': 'H2O AutoML (1h)',
    'TPOT_1h': 'TPOT (1h)',
    'AutoWEKA_1h': 'Auto-WEKA (1h)',
    'AutoPilot_1h': 'AutoPilot (1h)',
    'autogluon_4h': 'AutoGluon (4h)',
    'GCPTables_4h': 'GCP-Tables (4h)',
    'autosklearn_4h': 'auto-sklearn (4h)',
    'H2OAutoML_4h': 'H2O AutoML (4h)',
    'TPOT_4h': 'TPOT (4h)',
    'AutoWEKA_4h': 'Auto-WEKA (4h)',
    'autogluon_8h': 'AutoGluon (8h)',
    'GCPTables_8h': 'GCP-Tables (8h)',
    'H2OAutoML_8h': 'H2O AutoML (8h)',
    'autosklearn_8h': 'auto-sklearn (8h)',
    'TPOT_8h': 'TPOT (8h)',
    'AutoWEKA_8h': 'Auto-WEKA (8h)',

    'autogluon_nostack_1h': 'NoMultiStack (1h)',
    'autogluon_nobag_1h': 'NoBag (1h)',
    'autogluon_norepeatbag_1h': 'NoRepeat (1h)',
    'autogluon_nonn_1h': 'NoNetwork (1h)',
    'autogluon_nostack_4h': 'NoMultiStack (4h)',
    'autogluon_nobag_4h': 'NoBag (4h)',
    'autogluon_norepeatbag_4h': 'NoRepeat (4h)',
    'autogluon_nonn_4h': 'NoNetwork (4h)',
}

systemnames_keys = list(SYSTEM_NAMES.keys())
for key in systemnames_keys:
    SYSTEM_NAMES['orig_'+key] = SYSTEM_NAMES[key] + ' (O)'

NOTIME_NAMES = {
    'autogluon_1h': 'AutoGluon',
    'GCPTables_1h': 'GCP-Tables',
    'autosklearn_1h': 'auto-sklearn',
    'H2OAutoML_1h': 'H2O AutoML',
    'TPOT_1h': 'TPOT',
    'AutoPilot_1h': 'AutoPilot',
    'AutoWEKA_1h': 'Auto-WEKA',
    'autogluon_4h': 'AutoGluon',
    'GCPTables_4h': 'GCP-Tables',
    'autosklearn_4h': 'auto-sklearn',
    'H2OAutoML_4h': 'H2O AutoML',
    'TPOT_4h': 'TPOT',
    'AutoWEKA_4h': 'Auto-WEKA',
    'autogluon_8h': 'AutoGluon',
    'GCPTables_8h': 'GCP-Tables',
    'H2OAutoML_8h': 'H2O AutoML',
    'autosklearn_8h': 'auto-sklearn',
    'TPOT_8h': 'TPOT',
    'AutoWEKA_8h': 'Auto-WEKA',

    'autogluon_nostack_1h': 'NoMultiStack',
    'autogluon_nobag_1h': 'NoBag',
    'autogluon_norepeatbag_1h': 'NoRepeat',
    'autogluon_nonn_1h': 'NoNetwork',
    'autogluon_nostack_4h': 'NoMultiStack',
    'autogluon_nobag_4h': 'NoBag',
    'autogluon_norepeatbag_4h': 'NoRepeat',
    'autogluon_nonn_4h': 'NoNetwork',
}

notime_keys = list(NOTIME_NAMES.keys())
for key in notime_keys:
    NOTIME_NAMES['orig_'+key] = NOTIME_NAMES[key] + ' (O)'

KAGGLE_ABBREVS = {
    'house-prices-advanced-regression-techniques': 'house',
    'mercedes-benz-greener-manufacturing': 'mercedes', 
    'santander-value-prediction-challenge': 'value',
    'allstate-claims-severity': 'allstate', 
    'bnp-paribas-cardif-claims-management': 'bnp-paribas', 
    'santander-customer-transaction-prediction': 'transaction', 
    'santander-customer-satisfaction': 'satisfaction',
    'porto-seguro-safe-driver-prediction': 'porto', 'ieee-fraud-detection': 'ieee-fraud', 
    'walmart-recruiting-trip-type-classification': 'walmart',
    'otto-group-product-classification-challenge' : 'otto',
}

