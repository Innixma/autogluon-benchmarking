import pandas as pd
import os

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd

from autogluon_utils.benchmarking.evaluation import evaluate_results
from autogluon_utils.benchmarking.evaluation import generate_charts
from autogluon_utils.benchmarking.evaluation import tex_table
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'input/prepared/kaggle/'
    output_prefix = 'output/kaggle/'
    raw_kaggle_file = 'results_kaggle_wpercentile.csv'
    
    results_raw = load_pd.load(
        path=[
            results_dir_input + 'kaggle_core.csv',
        ]
    )
    # First generate datasets x frameworks raw data dumps:
    metrics = ['LEADER_PERCENTILE', METRIC_SCORE]
    dataset_order = ['house-prices-advanced-regression-techniques',
    'mercedes-benz-greener-manufacturing', 'santander-value-prediction-challenge',
    'allstate-claims-severity', 'bnp-paribas-cardif-claims-management', 
    'santander-customer-transaction-prediction', 'santander-customer-satisfaction',
    'porto-seguro-safe-driver-prediction', 'ieee-fraud-detection', 'walmart-recruiting-trip-type-classification',
    'otto-group-product-classification-challenge'
    ]
    dataset_order = [KAGGLE_ABBREVS[dat] for dat in dataset_order]
    method_order = ['AutoWEKA', 'autosklearn','TPOT', 'H2OAutoML','GCPTables','autogluon']
    time_limits = ['4h','8h']
    results_raw2 = results_raw.drop(METRIC_ERROR, axis = 1).copy()
    results_raw2['LEADER_PERCENTILE'] = 1 - results_raw2['LEADER_PERCENTILE'] # convert to actual percentile
    results_raw2.rename(columns={'LEADER_PERCENTILE': METRIC_ERROR}, inplace=True)
    
    # loss_df = generate_charts.compute_dataset_framework_df(results_raw) # values = losses
    percentile_df = generate_charts.compute_dataset_framework_df(results_raw2) 
    for time_limit in time_limits:
        methods_t = [meth + "_"+ time_limit for meth in method_order]
        df_time = percentile_df[[DATASET]+methods_t].copy()
        df_time[DATASET] = df_time[DATASET].map(KAGGLE_ABBREVS)
        df_ordered = df_time.set_index(DATASET)
        df_ordered = df_ordered.reindex(dataset_order)
        # df_ordered.reset_index(inplace=True)
        # df_ordered.rename(columns={'dataset': 'Dataset'},inplace=True)
        df_ordered.rename(columns=NOTIME_NAMES,inplace=True)
        save_pd.save(path=results_dir + output_prefix + time_limit + "/datasetsXframeworks.csv", df=df_ordered)
        textable_file = results_dir + output_prefix + time_limit + "/allpercentiles.tex"
        tex_table.tex_table(df_ordered,textable_file,bold = 'max',nan_char =" x ",max_digits=5)
    
    # Next do pairwise comparisons:
    num_frameworks = 6
    valid_frameworks = [
        'autogluon_4h',
        'GCPTables_4h',
        'autosklearn_4h',
        'H2OAutoML_4h',
        'TPOT_4h',
        'AutoWEKA_4h',
        'autogluon_8h',
        'GCPTables_8h',
        'H2OAutoML_8h',
        'autosklearn_8h',
        'TPOT_8h',
        'AutoWEKA_8h',
    ]
    
    frameworks_compare_vs_all_list = ['autogluon_4h','autogluon_8h', 'autogluon_4h', 'autogluon_8h']
    results_dir_output_list = ['4h/', '8h/', 'allVautogluon_4h/','allVautogluon_8h/']
    results_dir_output_list = [results_dir+output_prefix+name for name in results_dir_output_list]
    framework_compare_ind_list = [ # list of lists, each corresponding to indices of valid_frameworks that should be compared in a single table.
        list(range(num_frameworks)), list(range(num_frameworks,num_frameworks*2)), range(num_frameworks*2), range(num_frameworks*2), 
    ]
    
    for i in range(len(results_dir_output_list)):
        results_dir_output = results_dir_output_list[i]
        frameworks_to_compare = [valid_frameworks[j] for j in framework_compare_ind_list[i]]
        framework_compare_vs_all = frameworks_compare_vs_all_list[i]
        results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
            results_raw=results_raw,
            frameworks=frameworks_to_compare,
            banned_datasets=[],
            folds_to_keep=None,
            frameworks_compare_vs_all=[framework_compare_vs_all],
            output_dir=results_dir_output,
            columns_to_agg_extra=['LEADER_PERCENTILE'],
        )
        textab = tex_pairwise_table(results_dir_output, framework_compare_vs_all)

    # Generate plots:
    producePlots(time_limits, results_dir, raw_kaggle_file)


def producePlots(time_limits, results_dir, raw_kaggle_file):
    for time_limit in time_limits:
        print("Producing plots for time = %s" % time_limit)
        Rcode = "autogluon_utils/benchmarking/evaluation/plotting/kagglebenchmarkplots.R"
        command = " ".join(["Rscript --vanilla", Rcode, results_dir, raw_kaggle_file, time_limit])
        returned_value = os.system(command)
        print("R script exit code: %s" % returned_value)


# Generate tex kaggle pairwise results for paper:
def tex_pairwise_table(results_dir_output, framework_compare_vs_all, nan_char = " - "):
    pairwise_dir = results_dir_output + "pairwise/"
    pairwise_file = pairwise_dir + framework_compare_vs_all +".csv"
    pairwise_df = pd.read_csv(pairwise_file)
    main_framework_name = NOTIME_NAMES[framework_compare_vs_all]
    
    col_order = [FRAMEWORK, '> '+framework_compare_vs_all, '< '+framework_compare_vs_all, ERROR_COUNT, RANK_1, RANK, 'LEADER_PERCENTILE', TIME_TRAIN_S]
    df = pairwise_df[col_order].copy()
    df[FRAMEWORK] = df[FRAMEWORK].map(NOTIME_NAMES)
    df['LEADER_PERCENTILE'] = 1 - df['LEADER_PERCENTILE'] # convert to actual percentile
    df[TIME_TRAIN_S] =  df[TIME_TRAIN_S] / 60.0 # convert sec -> minutes
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S].round()
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S].astype(str)
    df[TIME_TRAIN_S] = df[TIME_TRAIN_S].str.replace('.0', '', regex=False)
    new_cols = ['Framework', 'Wins',  'Losses', 'Failures', 'Champion', 'Avg. Rank', 'Avg. Percentile', 'Avg. Time (min)']

    df.columns = ['\\textbf{'+col+'}' for col in new_cols]
    textable_file = pairwise_dir +framework_compare_vs_all + ".tex"
    df = df.replace("nan", nan_char)
    textab = df.to_latex(escape=False, index=False, na_rep=nan_char, 
                         column_format = 'l'+'c'*(len(df.columns)-1), float_format="{:0.4f}".format)
    with open(textable_file,'w') as tf:
        tf.write(textab)
        print("saved tex table to: %s" % textable_file)
    return textab


if __name__ == '__main__':
    run()
