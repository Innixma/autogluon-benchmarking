from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.loaders.load_s3 import list_bucket_prefix_suffix_contains_s3
from autogluon.utils.tabular.utils import s3_utils
from autogluon.utils.tabular.utils.savers import save_pd


def aggregate(path_prefix: str, contains=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix='scores/results.csv', contains=contains)
    print(objects)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    print(paths_full)
    df = load_pd.load(paths_full)
    print(df)
    return df


def aggregate_from_params(s3_bucket, s3_prefix, version_name, suffix, contains):
    result_path = s3_prefix + version_name + '/'
    aggregated_results_name = 'results_automlbenchmark' + suffix + '_' + version_name + '.csv'

    df = aggregate(path_prefix='s3://' + s3_bucket + '/results/' + result_path, contains=contains)

    save_pd.save(path='s3://' + s3_bucket + '/aggregated/' + result_path + aggregated_results_name, df=df)


if __name__ == '__main__':
    pass
    # ANONYMIZED AGGREGATION CODE
