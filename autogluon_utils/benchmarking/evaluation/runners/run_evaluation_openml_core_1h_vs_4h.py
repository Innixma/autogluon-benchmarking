import pandas as pd

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation import evaluate_results
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/prepared/openml/'
    results_dir_output = results_dir + 'output/openml/core_1h_vs_4h/'

    results_raw = load_pd.load(path=results_dir_input + 'openml_core.csv')

    frameworks = [
        'autogluon',
        'GCPTables',
        'H2OAutoML',
        'autosklearn',
        'TPOT',
        'AutoWEKA',
    ]

    folds_to_keep = [0]
    banned_datasets = []
    full_results_pairs_merged_dict = {}
    for framework in frameworks:
        run_path_prefix = framework + '/'
        framework_1h = framework + '_1h'
        framework_4h = framework + '_4h'

        results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
            results_raw=results_raw,
            frameworks=[framework_1h, framework_4h],
            banned_datasets=banned_datasets,
            folds_to_keep=folds_to_keep,
            columns_to_agg_extra=[
                # TIME_INFER_S,
                'acc',
                'auc',
                'logloss'
            ],
            frameworks_compare_vs_all=[framework_4h],
            output_dir=results_dir_output + run_path_prefix,
        )
        full_results_pairs_merged_dict.update(results_pairs_merged_dict)

    dfs = []
    for framework in frameworks:
        framework_1h = framework + '_1h'
        framework_4h = framework + '_4h'
        cur_df = full_results_pairs_merged_dict[framework_4h]
        cur_df = cur_df[cur_df[FRAMEWORK] == framework_1h]
        cur_columns = list(cur_df.columns)
        cur_columns[1] = '> 4h'
        cur_columns[2] = '< 4h'
        cur_columns[3] = '= 4h'
        cur_df.columns = cur_columns
        dfs.append(cur_df)
    df_final = pd.concat(dfs, ignore_index=True)
    print(df_final)
    save_pd.save(path=results_dir_output + 'pairwise/1h_vs_4h.csv', df=df_final)


if __name__ == '__main__':
    run()
