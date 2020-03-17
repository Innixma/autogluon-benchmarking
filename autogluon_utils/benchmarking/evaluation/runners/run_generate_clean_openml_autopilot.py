import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation.preprocess import preprocess_openml
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/raw/'
    results_dir_output = results_dir + 'input/prepared/openml/'

    ag_results_autopilot_1h = preprocess_openml.preprocess_openml_input(path=results_dir_input + 'results_automlbenchmark_autopilot_1h.csv', framework_suffix='_1h')

    results_list = [ag_results_autopilot_1h]
    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    frameworks_autopilot = ['AutoPilot_1h']

    results_core = results_raw[results_raw[FRAMEWORK].isin(frameworks_autopilot)]
    save_pd.save(path=results_dir_output + 'openml_autopilot.csv', df=results_core)


if __name__ == '__main__':
    run()
