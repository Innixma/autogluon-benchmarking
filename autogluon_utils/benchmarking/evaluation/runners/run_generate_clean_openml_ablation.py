import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation.preprocess import preprocess_openml
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/raw/'
    results_dir_output = results_dir + 'input/prepared/openml/'

    ag_results_ablation_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_ablation_1h.csv', framework_suffix='_1h')
    ag_results_ablation_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_ablation_4h.csv', framework_suffix='_4h')

    results_list = [ag_results_ablation_1h, ag_results_ablation_4h, ]
    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    frameworks_ablation = [
        'autogluon_nostack_1h',
        'autogluon_nobag_1h',
        'autogluon_norepeatbag_1h',
        'autogluon_nonn_1h',
        'autogluon_noknn_1h',

        'autogluon_nostack_4h',
        'autogluon_nobag_4h',
        'autogluon_norepeatbag_4h',
        'autogluon_nonn_4h',
        'autogluon_noknn_4h',
    ]

    results_ablation = results_raw[results_raw[FRAMEWORK].isin(frameworks_ablation)]
    save_pd.save(path=results_dir_output + 'openml_autogluon_ablation.csv', df=results_ablation)


if __name__ == '__main__':
    run()
