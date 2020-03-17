import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation.preprocess import preprocess_openml
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/raw/'
    results_dir_output = results_dir + 'input/prepared/openml/'

    ag_results_distilled_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_ag_leaderboard_1h_v15_distill.csv', framework_suffix='_1h')
    ag_results_distilled_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_ag_leaderboard_4h_v15_distill.csv', framework_suffix='_4h')

    results_list = [ag_results_distilled_1h, ag_results_distilled_4h, ]
    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    frameworks_distilled = [
        'autogluon_compressed_1h',
        'autogluon_distilled_1h',
        'autogluon_ensemble_1h',
        'autogluon_compressed_4h',
        'autogluon_distilled_4h',
        'autogluon_ensemble_4h',
    ]

    results_ablation = results_raw[results_raw[FRAMEWORK].isin(frameworks_distilled)]
    save_pd.save(path=results_dir_output + 'openml_autogluon_distilled.csv', df=results_ablation)


if __name__ == '__main__':
    run()
