import pandas as pd
import numpy as np
import logging

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd
import autogluon.utils.tabular.metrics as metrics
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

log = logging.getLogger(__name__)


def prepare_data(config, dataset):
    print('#################')
    print('Config:')
    print(config.__json__())
    print()
    print('Dataset:')
    print(dataset.__dict__)
    print('#################')

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    # un = dataset.train.path
    # print(un)
    # raw_data = loadarff(un)
    # df_data = pd.DataFrame(raw_data[0])

    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X
    y_test = dataset.test.y

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # Save and load data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv
    save_pd.save(path='tmp/tmp_file_train.csv', df=X_train)
    X_train = load_pd.load(path='tmp/tmp_file_train.csv')
    save_pd.save(path='tmp/tmp_file_test.csv', df=X_test)
    X_test = load_pd.load(path='tmp/tmp_file_test.csv')

    is_classification = config.type == 'classification'
    if is_classification:
        unique_vals = np.unique(y_train)
        if len(unique_vals) == 2:
            problem_type = BINARY
        else:
            problem_type = MULTICLASS
    else:
        problem_type = REGRESSION

    return X_train, y_train, X_test, y_test, problem_type, perf_metric
