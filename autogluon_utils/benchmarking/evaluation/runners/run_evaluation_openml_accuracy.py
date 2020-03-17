from autogluon.utils.tabular.utils.loaders import load_pd

from autogluon_utils.benchmarking.evaluation import evaluate_results
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/prepared/openml/'
    results_dir_output = results_dir + 'output/openml/accuracy/'

    results_raw = load_pd.load(
        path=[
            results_dir_input + 'openml_core.csv',
            results_dir_input + 'openml_autopilot.csv'
        ],
        worker_count=1
    )

    valid_frameworks = [
        'autogluon_1h',
        'GCPTables_1h',
        'H2OAutoML_1h',
        'autosklearn_1h',
        'TPOT_1h',
        'AutoWEKA_1h',
        'AutoPilot_1h',
    ]

    results_raw[METRIC_SCORE] = results_raw['acc']
    results_raw[METRIC_ERROR] = 1 - results_raw[METRIC_SCORE]
    run_path_prefix = '1h/'

    banned_datasets = []

    folds_to_keep = [0]
    results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=valid_frameworks,
        banned_datasets=banned_datasets,
        folds_to_keep=folds_to_keep,
        columns_to_agg_extra=[
            # TIME_INFER_S,
            'acc',
        ],
        frameworks_compare_vs_all=['autogluon_1h', 'AutoPilot_1h'],
        output_dir=results_dir_output + run_path_prefix,
    )


if __name__ == '__main__':
    run()
