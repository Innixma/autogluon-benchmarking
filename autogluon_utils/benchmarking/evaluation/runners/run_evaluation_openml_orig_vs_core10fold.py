import pandas as pd

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation import evaluate_results
from autogluon_utils.benchmarking.evaluation.constants_datasets import DATASETS_LARGE
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/prepared/openml/'
    results_dir_output = results_dir + 'output/openml/orig_vs_core10fold/'

    results_raw = load_pd.load(path=[
        results_dir_input + 'openml_core.csv',
        results_dir_input + 'openml_original.csv',
    ])

    frameworks_1h = [
        'H2OAutoML_1h',
        'autosklearn_1h',
        'TPOT_1h',
        'AutoWEKA_1h',
    ]

    frameworks_4h = [
        'H2OAutoML_4h',
        'autosklearn_4h',
        'TPOT_4h',
        'AutoWEKA_4h',
    ]

    frameworks_run_list = [frameworks_1h, frameworks_4h]
    folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    folds_to_keep_list = [folds, folds]
    banned_datasets_list = [DATASETS_LARGE, []]
    num_runs = len(frameworks_run_list)
    full_results_pairs_merged_dict = {}
    for i in range(num_runs):
        frameworks_run = frameworks_run_list[i]
        folds_to_keep = folds_to_keep_list[i]
        banned_datasets = banned_datasets_list[i]

        for framework in frameworks_run:
            run_path_prefix = framework + '/'
            orig_framework = 'orig_' + framework

            results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
                results_raw=results_raw,
                frameworks=[framework, orig_framework],
                banned_datasets=banned_datasets,
                folds_to_keep=folds_to_keep,
                columns_to_agg_extra=[
                    # TIME_INFER_S,
                    'acc',
                    'auc',
                    'logloss'
                ],
                frameworks_compare_vs_all=[orig_framework],
                output_dir=results_dir_output + run_path_prefix,
            )
            full_results_pairs_merged_dict.update(results_pairs_merged_dict)

    dfs = []
    frameworks_full = frameworks_1h + frameworks_4h
    for framework in frameworks_full:
        orig_framework = 'orig_' + framework
        cur_df = full_results_pairs_merged_dict[orig_framework]
        cur_df = cur_df[cur_df[FRAMEWORK] == framework]
        cur_columns = list(cur_df.columns)
        cur_columns[1] = '> Original'
        cur_columns[2] = '< Original'
        cur_columns[3] = '= Original'
        cur_df.columns = cur_columns
        dfs.append(cur_df)
    df_final = pd.concat(dfs, ignore_index=True)
    print(df_final)
    save_pd.save(path=results_dir_output + 'pairwise/new_vs_old.csv', df=df_final)


if __name__ == '__main__':
    run()
