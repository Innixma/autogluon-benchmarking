import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation.preprocess import preprocess_kaggle
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/raw/'
    results_dir_output = results_dir + 'input/prepared/kaggle/'

    kaggle_results = preprocess_kaggle.preprocess_kaggle_input(path=results_dir_input + 'results_kaggle_wpercentile.csv', framework_suffix='')

    kaggle_results[FRAMEWORK] = kaggle_results[FRAMEWORK].str.replace('GoogleAutoMLTables_', 'GCPTables_', regex=False)

    frameworks_core = [
        'autogluon_4h',
        'GCPTables_4h',
        'autosklearn_4h',
        'H2OAutoML_4h',
        'TPOT_4h',
        'AutoWEKA_4h',

        'autogluon_8h',
        'GCPTables_8h',
        'H2OAutoML_8h',
        'autosklearn_8h',
        'TPOT_8h',
        'AutoWEKA_8h',
    ]

    results_list = [kaggle_results]
    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    results_ablation = results_raw[results_raw[FRAMEWORK].isin(frameworks_core)]
    save_pd.save(path=results_dir_output + 'kaggle_core.csv', df=results_ablation)


if __name__ == '__main__':
    run()
