import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation.preprocess import preprocess_openml
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/raw/'
    results_dir_output = results_dir + 'input/prepared/openml/'

    ag_results_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_1h.csv', framework_suffix='_1h')
    ag_results_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_4h.csv', framework_suffix='_4h')

    ag_results_1h[FRAMEWORK] = ag_results_1h[FRAMEWORK].str.replace('_benchmark_', '_', regex=False)
    ag_results_4h[FRAMEWORK] = ag_results_4h[FRAMEWORK].str.replace('_benchmark_', '_', regex=False)

    gcp_results_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_gcptables_1h.csv', framework_suffix='_1h')
    gcp_results_4h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_gcptables_4h.csv', framework_suffix='_4h')
    gcp_results_1h[FRAMEWORK] = gcp_results_1h[FRAMEWORK].str.replace('GoogleAutoMLTables_benchmark_', 'GCPTables_', regex=False)
    gcp_results_4h[FRAMEWORK] = gcp_results_4h[FRAMEWORK].str.replace('GoogleAutoMLTables_benchmark_', 'GCPTables_', regex=False)
    gcp_results_1h = gcp_results_1h[gcp_results_1h[FRAMEWORK] == 'GCPTables_1h']
    gcp_results_4h = gcp_results_4h[gcp_results_4h[FRAMEWORK] == 'GCPTables_4h']

    results_list = [gcp_results_1h, gcp_results_4h, ag_results_1h, ag_results_4h]
    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    frameworks_core = [
        'autogluon_1h',
        'GCPTables_1h',
        'H2OAutoML_1h',
        'autosklearn_1h',
        'TPOT_1h',
        'AutoWEKA_1h',

        'autogluon_4h',
        'GCPTables_4h',
        'H2OAutoML_4h',
        'autosklearn_4h',
        'TPOT_4h',
        'AutoWEKA_4h',
    ]

    results_core = results_raw[results_raw[FRAMEWORK].isin(frameworks_core)]
    save_pd.save(path=results_dir_output + 'openml_core.csv', df=results_core)


if __name__ == '__main__':
    run()
