import pandas as pd

from autogluon.utils.tabular.utils.savers import save_pd

from .constants import *
from . import evaluate_utils
from.preprocess import preprocess_utils


def evaluate(results_raw, frameworks=None, banned_datasets=None, folds_to_keep=None, columns_to_agg_extra=None, frameworks_compare_vs_all=None, output_dir=None):
    if frameworks is None:
        frameworks = sorted(list(results_raw[FRAMEWORK].unique()))
    if frameworks_compare_vs_all is None:
        frameworks_compare_vs_all = []
    if folds_to_keep is None:
        folds_to_keep = sorted(list(results_raw[FOLD].unique()))
    if banned_datasets is not None:
        results_raw = results_raw[~results_raw[DATASET].isin(banned_datasets)]

    total_datasets = sorted(results_raw[DATASET].unique())
    results_raw = preprocess_utils.clean_result(result_df=results_raw, folds_to_keep=folds_to_keep, remove_invalid=True)

    results_raw = results_raw[results_raw[FRAMEWORK].isin(frameworks)]

    # Calculate each frameworks errored datasets
    total_frameworks = results_raw[FRAMEWORK].unique()
    total_folds = results_raw[FOLD].unique()
    num_frameworks = len(total_frameworks)
    num_datasets = len(total_datasets)
    num_folds = len(total_folds)
    ideal_rows = num_folds * num_datasets * num_frameworks
    actual_rows = len(results_raw)
    errors = ideal_rows - actual_rows
    print('num_datasets:', num_datasets)
    print('num_folds:', num_folds)
    print('errors:', errors)

    for framework in total_frameworks:
        results_framework = results_raw[results_raw[FRAMEWORK] == framework]
        num_rows_framework = len(results_framework)
        datasets_framework = results_framework[DATASET].unique()
        datasets_framework_errors = [dataset for dataset in total_datasets if dataset not in datasets_framework]
        datasets_framework_errors_count = len(datasets_framework_errors)
        framework_fold_errors = num_datasets * num_folds - num_rows_framework
        print('################################################')
        print('framework:', framework)
        print('datasets_framework_errors:', datasets_framework_errors)
        print('datasets_framework_errors_count:', datasets_framework_errors_count)
        print('framework_fold_errors:', framework_fold_errors)
        print('################################################')

    all_results_pairs = {}
    for framework_2 in frameworks_compare_vs_all:
        results_list = []

        for framework_1 in total_frameworks:
            if framework_1 == framework_2:
                results_ranked, results_ranked_by_dataset = evaluate_utils.compare_frameworks(results_raw=results_raw, frameworks=[framework_2], banned_datasets=banned_datasets, folds_to_keep=folds_to_keep, columns_to_agg_extra=columns_to_agg_extra, datasets=total_datasets, verbose=False)
                ties = len(results_ranked_by_dataset)
                results_list.append([framework_1, 0, 0, ties])
                continue

            results_ranked, results_ranked_by_dataset = evaluate_utils.compare_frameworks(results_raw=results_raw, frameworks=[framework_1, framework_2], banned_datasets=banned_datasets, folds_to_keep=folds_to_keep, columns_to_agg_extra=columns_to_agg_extra, datasets=total_datasets, verbose=False)

            datasets_pair = results_ranked_by_dataset[DATASET].unique()
            framework_1_wins = 0
            framework_2_wins = 0
            ties = 0
            for dataset in datasets_pair:
                results_isolated = results_ranked_by_dataset[results_ranked_by_dataset[DATASET] == dataset]
                results_isolated = results_isolated[results_isolated[FRAMEWORK] == framework_1]
                results_isolated_rank = results_isolated[RANK].iloc[0]
                if results_isolated_rank == 1:
                    framework_1_wins += 1
                elif results_isolated_rank == 2:
                    framework_2_wins += 1
                elif results_isolated_rank == 1.5:
                    ties += 1
                else:
                    raise AssertionError('Rank not valid: %s' % results_isolated_rank)
            results_list.append([framework_1, framework_1_wins, framework_2_wins, ties])
        results_pairs = pd.DataFrame(data=results_list, columns=[FRAMEWORK, '> ' + framework_2, '< ' + framework_2, '= ' + framework_2])
        all_results_pairs[framework_2] = results_pairs

    print('################################################')
    print('%s VS %s' % ('all', 'all'))
    print('\tAll datasets regardless of failures')
    results_ranked_all, results_ranked_by_dataset_all = evaluate_utils.compare_frameworks(results_raw=results_raw, banned_datasets=banned_datasets, folds_to_keep=folds_to_keep, filter_errors=False, columns_to_agg_extra=columns_to_agg_extra, datasets=total_datasets)

    if output_dir:
        save_pd.save(path=output_dir + 'results_ranked_all.csv', df=results_ranked_all)
        save_pd.save(path=output_dir + 'results_ranked_by_dataset_all.csv', df=results_ranked_by_dataset_all)

    print('################################################')
    print('%s VS %s' % ('all', 'all'))
    print('\tOnly datasets where all frameworks succeeded')
    results_ranked_valid, results_ranked_by_dataset_valid = evaluate_utils.compare_frameworks(results_raw=results_raw, frameworks=frameworks, banned_datasets=banned_datasets, folds_to_keep=folds_to_keep, columns_to_agg_extra=columns_to_agg_extra, datasets=total_datasets)

    results_pairs_merged_dict = {}
    for framework in frameworks_compare_vs_all:
        columns_to_get_from_all = [RANK_1, 'rank=2_count', 'rank=3_count', 'rank>3_count', ERROR_COUNT]
        results_pairs = all_results_pairs[framework]
        results_pairs_merged = pd.merge(results_pairs, results_ranked_valid, on=FRAMEWORK, how='left')
        results_pairs_merged = results_pairs_merged.drop(columns_to_get_from_all, axis=1)
        results_pairs_merged = pd.merge(results_pairs_merged, results_ranked_all[[FRAMEWORK] + columns_to_get_from_all], on=FRAMEWORK, how='left')
        results_pairs_merged = results_pairs_merged.sort_values(by=RANK)
        print('################################################')
        print('%s VS %s' % (framework, 'all'))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_pairs_merged)
        if output_dir:
            save_pd.save(path=output_dir + 'pairwise/' + framework + '.csv', df=results_pairs_merged)
        results_pairs_merged_dict[framework] = results_pairs_merged

    if output_dir:
        save_pd.save(path=output_dir + 'results_ranked_valid.csv', df=results_ranked_valid)
        save_pd.save(path=output_dir + 'results_ranked_by_dataset_valid.csv', df=results_ranked_by_dataset_valid)

    return results_ranked_valid, results_ranked_by_dataset_valid, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict
