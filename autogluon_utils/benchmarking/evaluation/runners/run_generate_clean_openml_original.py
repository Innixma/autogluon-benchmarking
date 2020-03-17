import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation.preprocess import preprocess_openml
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/raw/original/'
    results_dir_output = results_dir + 'input/prepared/openml/'

    other_results_large_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_large-8c4h.csv', framework_suffix='_4h')
    other_results_medium_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_medium-8c4h.csv', framework_suffix='_4h')
    other_results_small_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_small-8c4h.csv', framework_suffix='_4h')
    other_results_medium_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_medium-8c1h.csv', framework_suffix='_1h')
    other_results_small_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_small-8c1h.csv', framework_suffix='_1h')

    results_list = [other_results_large_4h, other_results_medium_4h, other_results_small_4h, other_results_medium_1h, other_results_small_1h]

    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    results_raw[FRAMEWORK] = ['orig_' + name[0] for name in zip(results_raw[FRAMEWORK])]

    frameworks_original = [
        'orig_H2OAutoML_1h',
        'orig_autosklearn_1h',
        'orig_TPOT_1h',
        'orig_AutoWEKA_1h',

        'orig_H2OAutoML_4h',
        'orig_autosklearn_4h',
        'orig_TPOT_4h',
        'orig_AutoWEKA_4h',
    ]

    results_original = results_raw[results_raw[FRAMEWORK].isin(frameworks_original)]
    save_pd.save(path=results_dir_output + 'openml_original.csv', df=results_original)


if __name__ == '__main__':
    run()
