import numpy as np
import pandas as pd

from autogluon_utils.benchmarking.evaluation.constants import *


def compute_dataset_framework_df(results_df):
    results_df = results_df.copy()
    df = results_df[[FRAMEWORK, DATASET, METRIC_ERROR]].copy()

    datasets = sorted(list(df[DATASET].unique()))
    frameworks = sorted(list(df[FRAMEWORK].unique()))

    df_new = []
    for dataset in datasets:
        row = [dataset]
        df_dataset = df[df[DATASET] == dataset]
        for framework in frameworks:
            df_metric_error = df_dataset[df_dataset[FRAMEWORK] == framework].reset_index(drop=True)
            if len(df_metric_error) == 0:
                row.append(np.nan)
            else:
                row.append(df_metric_error[METRIC_ERROR][0])
        df_new.append(row)
    df_result = pd.DataFrame(data=df_new, columns=[DATASET] + frameworks)

    return df_result


if __name__ == '__main__':
    results_df = pd.DataFrame(data={
        DATASET: ['a', 'a', 'a', 'c'],
        FRAMEWORK: ['ag', 'h2o', 'tab', 'ag'],
        METRIC_ERROR: [0.4, 0.2, 0.6, 0.7],
    })
    print(results_df)

    result = compute_dataset_framework_df(results_df)

    print(result)
