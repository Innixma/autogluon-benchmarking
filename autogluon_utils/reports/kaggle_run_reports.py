import boto3
import fire
import pandas as pd
from pandas import DataFrame

from autogluon_utils.utils.config import DEFAULT_CONFIG

__RANK = '__rank__'


def generate_kaggle_run_reports(save_to_path: str = None, 
                                rank_top_for_column: str = None, ascending=False,
                                bucket: str = None, prefix: str = None, 
                                session: boto3.session.Session = None) -> DataFrame:
    """
    :param save_to_path: location to save the output dataframe to
    :param ascending: if rank_top_for_column used, then use ordering as specified in this parameter
    :param rank_top_for_column: which column to use for getting top results
    :param bucket: bucket of the location where competition data is stored; will use DEFAULT_CONFIG if not specified
    :param prefix: prefix of the location where competition data is stored; will use DEFAULT_CONFIG if not specified
    :param session: Optional boto3 session can be supplied use your own credentials rather than the defaults.
    :return: aggregated run reports data from metrics.json files
    """
    if bucket is None:
        path = DEFAULT_CONFIG['s3_path']
        split_pos = path.find('/')
        bucket, prefix = path[:split_pos], path[split_pos + 1:]

    if session is None:
        client = boto3.client('s3')
    else:
        client = session.client('s3')
    
    paginator = client.get_paginator('list_objects')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    all_metrics = []
    for page in pages:
        for entry in page['Contents']:
            key = entry['Key']

            if key.endswith('metrics.json'):
                _metrics_file_handler(all_metrics, bucket, key)

            elif key.endswith('exception.txt'):
                _exceptions_file_handler(all_metrics, bucket, key)

    results = pd.concat(all_metrics, sort=True).reset_index(drop=True)

    if rank_top_for_column is not None:
        print(f'Grouping outputs by competition and ranking by {rank_top_for_column}, ascending={ascending}')
        results: DataFrame = _get_top_result(results, rank_top_for_column, ascending)

    results = reorder_columns(results) # Reorder columns to faciliate interpretation
    if save_to_path is not None:
        results.to_csv(save_to_path, index=False)
        print(f'Saved results to {save_to_path}')

    return results


def _exceptions_file_handler(all_metrics, bucket, key):
    key_splits = key.split('/')
    if len(key_splits) == 8:  # old format without tags
        _, competition, _, date, profile, predictor, _, file = key_splits
        tag = None
    elif len(key_splits) == 9:  # new format with tags
        _, competition, _, predictor, profile, tag, date, _, file = key_splits
    else:
        print(f'Unknown url format - skipping [{key}]')
        return

    file = '/'.join(['s3:/', bucket, key])
    try:
        # read each line as a separate value
        metrics = pd.read_csv(file, sep='!@#$%^!@#$%', engine='python')[-1:]
        metrics.columns = ['error_description']
        metrics = _apply_path_parameters(competition, date, file, metrics, predictor, profile, tag)
        all_metrics.append(metrics)
    except Exception as err:
        print(f'Error - skipping [{key}]')
        print(err)


def _metrics_file_handler(all_metrics, bucket, key):
    key_splits = key.split('/')
    if len(key_splits) == 7:  # old format without tags
        _, competition, _, date, profile, predictor, file = key_splits
        tag = None
    elif len(key_splits) == 8:  # new format with tags
        _, competition, _, predictor, profile, tag, date, file = key_splits
    else:
        print(f'Unknown url format - skipping [{key}]')
        return

    file = '/'.join(['s3:/', bucket, key])
    try:
        metrics = pd.read_json(file, lines=True)
        metrics = metrics.rename(columns={'date': 'kaggle_date'})
        metrics = _apply_path_parameters(competition, date, file, metrics, predictor, profile, tag)
        all_metrics.append(metrics)
    except Exception as err:
        print(f'Error - skipping [{key}]')
        print(err)


def _apply_path_parameters(competition, date, file, metrics, predictor, profile, tag):
    metrics['predictor'] = predictor
    metrics['profile'] = profile
    metrics['path'] = file[:file.rfind('/')]
    metrics['competition'] = competition
    metrics['tag'] = tag
    metrics['date'] = date
    metrics = metrics[sorted(metrics.columns)]
    return metrics


def _get_top_result(report: DataFrame, rank_column: str, ascending=False) -> DataFrame:
    report = report.copy()
    report[__RANK] = report.groupby('competition')[rank_column].rank("dense", ascending=ascending)
    return report[report[__RANK] == 1].drop(columns=__RANK)


def reorder_columns(results_df):
    col_names = results_df.columns.tolist()
    primary_cols = ['competition', 'predictor', 'private_score', 'public_score', 'rank',
                    'fit_time', 'pred_time', 'profile', 'num_models_trained']
    end_cols = ['date', 'error_description']
    specified_cols = primary_cols + end_cols
    unspecified_cols = [col for col in col_names if col not in specified_cols]
    new_col_order = primary_cols + unspecified_cols + end_cols
    results_df = results_df[new_col_order]
    return results_df

if __name__ == '__main__':
    fire.Fire(generate_kaggle_run_reports)  # CLI wrapper


