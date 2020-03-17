from autogluon.utils.tabular.utils.loaders import load_pd

from autogluon_utils.benchmarking.evaluation import evaluate_results


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/prepared/openml/'
    results_dir_output = results_dir + 'output/openml/core10fold/'

    results_raw = load_pd.load(path=results_dir_input + 'openml_core.csv')

    frameworks_1h = [
        'autogluon_1h',
        'H2OAutoML_1h',
        'autosklearn_1h',
        'TPOT_1h',
        'AutoWEKA_1h',
    ]

    frameworks_4h = [
        'autogluon_4h',
        'H2OAutoML_4h',
        'autosklearn_4h',
        'TPOT_4h',
        'AutoWEKA_4h',
    ]

    run_path_prefix_list = ['1h/', '4h/']
    frameworks_compare_vs_all_list = [['autogluon_1h'], ['autogluon_4h']]
    frameworks_run_list = [frameworks_1h, frameworks_4h]
    folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    folds_to_keep_list = [folds, folds]
    banned_datasets = []
    num_runs = len(run_path_prefix_list)
    for i in range(num_runs):
        run_path_prefix = run_path_prefix_list[i]
        frameworks_compare_vs_all = frameworks_compare_vs_all_list[i]
        frameworks_run = frameworks_run_list[i]
        folds_to_keep = folds_to_keep_list[i]

        results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
            results_raw=results_raw,
            frameworks=frameworks_run,
            banned_datasets=banned_datasets,
            folds_to_keep=folds_to_keep,
            columns_to_agg_extra=[
                # TIME_INFER_S,
                'acc',
                'auc',
                'logloss'
            ],
            frameworks_compare_vs_all=frameworks_compare_vs_all,
            output_dir=results_dir_output + run_path_prefix,
        )


if __name__ == '__main__':
    run()
