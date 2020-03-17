import os
import pandas as pd

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon_utils.benchmarking.evaluation import generate_charts
from autogluon_utils.benchmarking.evaluation import tex_table
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'

    results_dir_input = results_dir + 'output/openml/core/1h/'
    framework_compare_vs_all = 'autogluon_1h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)
    generate_tex_datasetXframework_table(results_dir_input, '1h')

    results_dir_input = results_dir + 'output/openml/core/4h/'
    framework_compare_vs_all = 'autogluon_4h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)
    generate_tex_datasetXframework_table(results_dir_input, '4h')

    results_dir_input = results_dir + 'output/openml/accuracy/1h/'
    method_order = ['AutoWEKA', 'autosklearn', 'TPOT', 'H2OAutoML', 'GCPTables', 'AutoPilot', 'autogluon']
    generate_tex_datasetXframework_table(results_dir_input, '1h', method_order=method_order)

    results_dir_input = results_dir + 'output/openml/ablation/1h/'
    framework_compare_vs_all = 'autogluon_1h'
    run_single(results_dir_input, framework_compare_vs_all, drop_columns=['Failures'], framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/ablation/4h/'
    framework_compare_vs_all = 'autogluon_4h'
    run_single(results_dir_input, framework_compare_vs_all, drop_columns=['Failures'], framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/ablation/4h/'
    framework_compare_vs_all = 'autogluon_4h'
    run_single(results_dir_input, framework_compare_vs_all, drop_columns=['Wins',  'Losses', 'Failures', 'Champion', 'Avg. Time (min)'], framework_name_map=NOTIME_NAMES, suffix='_mini')

    results_dir_input = results_dir + 'output/openml/ablation/combined/'
    framework_compare_vs_all = 'autogluon_1h'
    run_single(results_dir_input, framework_compare_vs_all, drop_columns=['Failures'], framework_name_map=SYSTEM_NAMES)

    results_dir_input = results_dir + 'output/openml/ablation/combined/'
    framework_compare_vs_all = 'autogluon_4h'
    run_single(results_dir_input, framework_compare_vs_all, drop_columns=['Failures'], framework_name_map=SYSTEM_NAMES)

    results_dir_input = results_dir + 'output/openml/accuracy/1h/'
    framework_compare_vs_all = 'autogluon_1h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/core10fold/1h/'
    framework_compare_vs_all = 'autogluon_1h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/core10fold/4h/'
    framework_compare_vs_all = 'autogluon_4h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/orig10fold/1h/'
    framework_compare_vs_all = 'autogluon_1h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/orig10fold/4h/'
    framework_compare_vs_all = 'autogluon_4h'
    run_single(results_dir_input, framework_compare_vs_all, framework_name_map=NOTIME_NAMES)

    results_dir_input = results_dir + 'output/openml/core_1h_vs_4h/'
    run_single_vs(results_dir_input, filename='1h_vs_4h', col_name_comparison_str='4h', framework_name_map=SYSTEM_NAMES)

    results_dir_input = results_dir + 'output/openml/orig_vs_core10fold/'
    run_single_vs(results_dir_input, filename='new_vs_old', col_name_comparison_str='Original', framework_name_map=SYSTEM_NAMES)


def run_single(results_dir_input, framework_compare_vs_all, drop_columns=None, framework_name_map=SYSTEM_NAMES, suffix=''):
    input_openml = results_dir_input + 'pairwise/' + framework_compare_vs_all + '.csv'
    results_dir_output = results_dir_input + 'tex/'
    pairwise_df = load_pd.load(input_openml)
    textab = generate_tex_pairwise_table(pairwise_df=pairwise_df, framework_compare_vs_all=framework_compare_vs_all, drop_columns=drop_columns, framework_name_map=framework_name_map)
    textable_file = results_dir_output + 'pairwise/' + framework_compare_vs_all + suffix + ".tex"
    os.makedirs(os.path.dirname(textable_file), exist_ok=True)
    with open(textable_file, 'w') as tf:
        tf.write(textab)
        print("saved tex table to: %s" % textable_file)


def run_single_vs(results_dir_input, filename, col_name_comparison_str, framework_name_map=SYSTEM_NAMES):
    results_dir_output = results_dir_input + 'tex/'
    pairwise_vs_df = load_pd.load(results_dir_input + 'pairwise/' + filename + '.csv')
    textable_file = results_dir_output + 'pairwise/' + filename + ".tex"
    textab = generate_tex_pairwise_vs_table(pairwise_vs_df, col_name_comparison_str=col_name_comparison_str, framework_name_map=framework_name_map)
    os.makedirs(os.path.dirname(textable_file), exist_ok=True)
    with open(textable_file, 'w') as tf:
        tf.write(textab)
        print("saved tex table to: %s" % textable_file)


# Generate tex kaggle pairwise vs results for paper:
def generate_tex_pairwise_vs_table(pairwise_vs_df, col_name_comparison_str, nan_char=" - ", framework_name_map=SYSTEM_NAMES):
    col_order = [FRAMEWORK, '> ' + col_name_comparison_str, '< ' + col_name_comparison_str, '= ' + col_name_comparison_str]
    df = pairwise_vs_df[col_order].copy()
    df[FRAMEWORK] = df[FRAMEWORK].map(framework_name_map)
    new_cols = ['System', '$>$ ' + col_name_comparison_str, '$<$ ' + col_name_comparison_str, '$=$ ' + col_name_comparison_str]
    df.columns = ['\\textbf{' + col + '}' for col in new_cols]
    df = df.replace("nan", nan_char)
    textab = df.to_latex(escape=False, index=False, na_rep=nan_char,
                         column_format='l' + 'c' * (len(df.columns) - 1), float_format="{:0.4f}".format)
    return textab


# Generate tex kaggle pairwise results for paper:
def generate_tex_pairwise_table(pairwise_df, framework_compare_vs_all, nan_char=" - ", drop_columns=None, framework_name_map=SYSTEM_NAMES):
    main_framework_name = framework_name_map[framework_compare_vs_all]

    col_order = [FRAMEWORK, '> ' + framework_compare_vs_all, '< ' + framework_compare_vs_all, ERROR_COUNT, RANK_1, RANK, LOSS_RESCALED, TIME_TRAIN_S]
    df = pairwise_df[col_order].copy()
    df[FRAMEWORK] = df[FRAMEWORK].map(framework_name_map)
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S] / 60.0  # convert sec -> minutes
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S].round()
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S].astype(str)
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S].str.replace('.0', '', regex=False)
    new_cols = ['Framework', 'Wins',  'Losses', 'Failures', 'Champion', 'Avg. Rank', 'Avg. Rescaled Loss', 'Avg. Time (min)']
    df.columns = new_cols
    if drop_columns is not None:
        df = df.drop(columns=drop_columns)

    df.columns = ['\\textbf{' + col + '}' for col in list(df.columns)]
    df = df.replace("nan", nan_char)
    textab = df.to_latex(escape=False, index=False, na_rep=nan_char,
                         column_format='l' + 'c' * (len(df.columns) - 1), float_format="{:0.4f}".format)
    return textab


def generate_tex_datasetXframework_table(results_dir_input, time_limit, method_order=None):
    """ # generate datasets x frameworks raw data dumps """
    results_dir_output = results_dir_input + 'tex/'
    results_raw = load_pd.load(
        path=[
            results_dir_input + 'results_ranked_by_dataset_all.csv',
        ]
    )
    if method_order is None:
        method_order = ['AutoWEKA', 'autosklearn','TPOT', 'H2OAutoML','GCPTables','autogluon']
    metric_error_df = generate_charts.compute_dataset_framework_df(results_raw)
    print("metric_error_df:")
    print(metric_error_df.head())
    metric_error_df[DATASET] = pd.Series([x[:17] for x in list(metric_error_df[DATASET])])
    df_ordered = metric_error_df.set_index(DATASET)
    df_ordered = df_ordered[[meth+"_"+time_limit for meth in method_order]].copy()
    df_ordered.rename(columns={'dataset': 'Dataset'},inplace=True)
    df_ordered.rename(columns=NOTIME_NAMES,inplace=True)
    # save_pd.save(path=results_dir_output + "openml_datasetsXframeworks_"+time_limit+".csv", df=df_ordered)
    textable_file = results_dir_output + "openml_alllosses_"+time_limit+".tex"
    if not os.path.exists(results_dir_output):
        os.makedirs(results_dir_output)

    tex_table.tex_table(df_ordered,textable_file,bold = 'min',nan_char =" x ",max_digits=5)


if __name__ == '__main__':
    run()
