import pandas as pd

from .constants import *


def compare_frameworks(results_raw, frameworks=None, banned_datasets=None, folds_to_keep=None, filter_errors=True, verbose=True, columns_to_agg_extra=None, datasets=None):
    columns_to_agg = [DATASET, FRAMEWORK, PROBLEM_TYPE, TIME_TRAIN_S, METRIC_ERROR]
    if columns_to_agg_extra:
        columns_to_agg += columns_to_agg_extra
    if frameworks is None:
        frameworks = sorted(list(results_raw[FRAMEWORK].unique()))

    if filter_errors:  # FIXME: This should not be toggled, instead filter_errors should be passed to filter_results
        results = filter_results(results_raw=results_raw, valid_frameworks=frameworks, banned_datasets=banned_datasets, folds_to_keep=folds_to_keep)
    else:
        results = results_raw.copy()

    results_agg = results[columns_to_agg].groupby([DATASET, FRAMEWORK, PROBLEM_TYPE]).mean().reset_index()

    worst_scores = results_agg.sort_values(METRIC_ERROR, ascending=False).drop_duplicates(DATASET)
    worst_scores = worst_scores[[DATASET, METRIC_ERROR]]
    worst_scores.columns = [DATASET, 'WORST_ERROR']
    best_scores = results_agg.sort_values(METRIC_ERROR, ascending=True).drop_duplicates(DATASET)
    best_scores = best_scores[[DATASET, METRIC_ERROR]]
    best_scores.columns = [DATASET, 'BEST_ERROR']

    results_agg = results_agg.merge(best_scores, on=DATASET)
    results_agg = results_agg.merge(worst_scores, on=DATASET)
    results_agg[BESTDIFF] = 1 - (results_agg['BEST_ERROR'] / results_agg[METRIC_ERROR])
    results_agg[LOSS_RESCALED] = (results_agg[METRIC_ERROR] - results_agg['BEST_ERROR']) / (results_agg['WORST_ERROR'] - results_agg['BEST_ERROR'])
    results_agg[BESTDIFF] = results_agg[BESTDIFF].fillna(0)
    results_agg[LOSS_RESCALED] = results_agg[LOSS_RESCALED].fillna(0)
    results_agg = results_agg.drop(['BEST_ERROR'], axis=1)
    results_agg = results_agg.drop(['WORST_ERROR'], axis=1)

    valid_tasks = list(results_agg[DATASET].unique())

    results_ranked, results_ranked_by_dataset = rank_result(results_agg)
    rank_1 = results_ranked_by_dataset[results_ranked_by_dataset[RANK] == 1]
    rank_1_count = rank_1[FRAMEWORK].value_counts()
    results_ranked['rank=1_count'] = rank_1_count
    results_ranked['rank=1_count'] = results_ranked['rank=1_count'].fillna(0).astype(int)

    rank_2 = results_ranked_by_dataset[(results_ranked_by_dataset[RANK] > 1) & (results_ranked_by_dataset[RANK] <= 2)]
    rank_2_count = rank_2[FRAMEWORK].value_counts()

    results_ranked['rank=2_count'] = rank_2_count
    results_ranked['rank=2_count'] = results_ranked['rank=2_count'].fillna(0).astype(int)

    rank_3 = results_ranked_by_dataset[(results_ranked_by_dataset[RANK] > 2) & (results_ranked_by_dataset[RANK] <= 3)]
    rank_3_count = rank_3[FRAMEWORK].value_counts()

    results_ranked['rank=3_count'] = rank_3_count
    results_ranked['rank=3_count'] = results_ranked['rank=3_count'].fillna(0).astype(int)

    rank_l3 = results_ranked_by_dataset[(results_ranked_by_dataset[RANK] > 3)]
    rank_l3_count = rank_l3[FRAMEWORK].value_counts()

    results_ranked['rank>3_count'] = rank_l3_count
    results_ranked['rank>3_count'] = results_ranked['rank>3_count'].fillna(0).astype(int)

    if datasets is None:
        datasets = sorted(list(results_ranked_by_dataset[DATASET].unique()))
    errors_list = []
    for framework in frameworks:
        results_framework = filter_results(results_raw=results_raw, valid_frameworks=[framework], banned_datasets=banned_datasets, folds_to_keep=folds_to_keep)
        results_framework_agg = results_framework[columns_to_agg].groupby([DATASET, FRAMEWORK, PROBLEM_TYPE]).mean().reset_index()
        num_valid = len(results_framework_agg[results_framework_agg[FRAMEWORK] == framework])
        num_errors = len(datasets) - num_valid
        errors_list.append(num_errors)
    errors_series = pd.Series(data=errors_list, index=frameworks)

    results_ranked['error_count'] = errors_series
    results_ranked['error_count'] = results_ranked['error_count'].fillna(0).astype(int)
    results_ranked = results_ranked.reset_index()

    if verbose:
        print('valid_tasks:', len(valid_tasks))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_ranked)
        print()
    return results_ranked, results_ranked_by_dataset


def filter_results(results_raw, valid_frameworks, banned_datasets=None, folds_to_keep=None):
    results = results_raw.copy()
    if folds_to_keep is not None:
        results = results[results[FOLD].isin(folds_to_keep)]

    results = keep_only_valid_datasets(results, valid_models=valid_frameworks)

    if banned_datasets is not None:
        results = results[~results[DATASET].isin(banned_datasets)]

    return results


def keep_only_valid_datasets(result_df, valid_models):
    tasks = list(result_df[DATASET].unique())
    frameworks = list(result_df[FRAMEWORK].unique())
    dfs = []
    for task in tasks:
        df = result_df[result_df[DATASET] == task]
        df = df[df[FRAMEWORK].isin(valid_models)]
        if set(list(df[FRAMEWORK].unique())) == set(valid_models):
            dfs.append(df)
        else:
            valid_models_found = list(df[FRAMEWORK].unique())
            models_not_found = [model for model in valid_models if model not in valid_models_found]
            # print('NOT ALL FRAMEWORKS HAVE RESULT FOR', task)
            # print('Required:', valid_models)
            # print('Found:   ', valid_models_found)
            # print('Missing: ', models_not_found)
    if len(dfs) == 0:
        valid_result_df = pd.DataFrame(columns=result_df.columns)
    else:
        valid_result_df = pd.concat(dfs, ignore_index=True)
    return valid_result_df


def rank_result(result_df):
    datasets = list(result_df[DATASET].unique())
    dfs = []
    for dataset in datasets:
        dataset_df = result_df[result_df[DATASET] == dataset].copy()
        dataset_df[METRIC_ERROR] = [round(x[0], 5) for x in zip(dataset_df[METRIC_ERROR])]
        sorted_df = dataset_df.sort_values(by=[METRIC_ERROR])
        sorted_df[RANK] = sorted_df[METRIC_ERROR].rank()
        dfs.append(sorted_df)
    sorted_df_full = pd.concat(dfs, ignore_index=True)
    model_ranks_df = sorted_df_full.groupby([FRAMEWORK]).mean().sort_values(by=RANK)
    return model_ranks_df, sorted_df_full
