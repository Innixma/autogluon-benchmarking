import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from autogluon_utils.benchmarking.evaluation.constants import *


def graph_percent_error_diff(df, model_1, model_2, save_path=None):
    df_only_required_models = df[df[FRAMEWORK].isin([model_1, model_2])]
    models = df_only_required_models[FRAMEWORK].unique()

    num_models = len(models)
    dataset_names = df_only_required_models[DATASET].unique()
    dfs = []
    for d in dataset_names:
        dataset_df = df_only_required_models[df_only_required_models[DATASET] == d].copy()
        dataset_df[METRIC_ERROR] = [round(x[0], 5) for x in zip(dataset_df[METRIC_ERROR])]
        sorted_df = dataset_df.sort_values(by=[METRIC_ERROR])
        if len(sorted_df) == num_models:
            dfs.append(sorted_df)
    sorted_df_full = pd.concat(dfs, ignore_index=True)

    sorted_df_full_req = sorted_df_full[[DATASET, FRAMEWORK, METRIC_ERROR, NUM_ROWS, NUM_COLS]]
    dataset_names = sorted_df_full_req[DATASET].unique()

    m1_errs = []
    m2_errs = []
    num_cols = []
    num_rows = []
    for dataset_name in dataset_names:
        m1_errs.append(sorted_df_full_req[(sorted_df_full_req[DATASET] == dataset_name) & (sorted_df_full_req[FRAMEWORK] == model_1)][METRIC_ERROR].iloc[0])
        m2_errs.append(sorted_df_full_req[(sorted_df_full_req[DATASET] == dataset_name) & (sorted_df_full_req[FRAMEWORK] == model_2)][METRIC_ERROR].iloc[0])
        num_rows.append(sorted_df_full_req[(sorted_df_full_req[DATASET] == dataset_name) & (sorted_df_full_req[FRAMEWORK] == model_1)][NUM_ROWS].iloc[0])
        num_cols.append(sorted_df_full_req[(sorted_df_full_req[DATASET] == dataset_name) & (sorted_df_full_req[FRAMEWORK] == model_1)][NUM_COLS].iloc[0])

    percent_error_diffs = []
    for m1_err, m2_err in zip(m1_errs, m2_errs):
        if m1_err == m2_err:
            diff_sign = 0
        else:
            if m1_err < m2_err:
                sign = -1
            else:
                sign = 1
            worst_error = max(m1_err, m2_err)
            best_error = min(m1_err, m2_err)
            diff = 1 - best_error / worst_error
            diff_sign = sign * diff
        percent_error_diffs.append(diff_sign*100)

    colors = []
    num_datasets = len(dataset_names)
    m1_wins = 0
    m2_wins = 0
    ties = 0
    for percent_error_diff in percent_error_diffs:
        if percent_error_diff == 0:
            colors.append('blue')
            ties += 1
        elif percent_error_diff > 0:
            colors.append('green')
            m2_wins += 1
        else:
            colors.append('red')
            m1_wins += 1

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(percent_error_diffs, num_rows, c=colors)
    # ax.plot((0, 1), 'r--', label='Random guess')
    plt.axvline(x=0, color='k', linestyle='--')
    for i, xy in enumerate(zip(percent_error_diffs, num_rows)):  # <--
        ax.annotate(dataset_names[i], xy=xy, textcoords='data')  # <--

    mean_diff = np.mean(percent_error_diffs)
    mean_diff = np.round(mean_diff, 4)
    if mean_diff > 0:
        winner_color = 'green'
    elif mean_diff == 0:
        winner_color = 'blue'
    else:
        winner_color = 'red'
    plt.axvline(x=mean_diff, color=winner_color, linestyle='--')

    ax.set_xlabel('Percent Error Difference')
    # ax.set_ylabel('Row Count')
    ax.set_ylabel('AutoGluon Runtime (s)')
    # plt.plot(x, y, 'o', color='black')
    ax.set_title(model_1 + ' vs ' + model_2 + ' (Percent Error Difference)')
    ax.set_xticks(np.arange(-100.1, 100.1, 10))
    ax.set_yscale('log')
    # ax.set_yticks(np.arange(0, num_rows + 1, 0.1))
    plt.tight_layout()
    plt.grid()

    green_patch = mpatches.Patch(color='red', label='Winner: ' + model_1 + ' (' + str(m1_wins) + '/' + str(num_datasets) + ')')
    red_patch = mpatches.Patch(color='green', label='Winner: ' + model_2 + ' (' + str(m2_wins) + '/' + str(num_datasets) + ')')
    blue_patch = mpatches.Patch(color='blue', label='Tie' + ' (' + str(ties) + '/' + str(num_datasets) + ')')
    winner_patch = Line2D([0], [0], color=winner_color, linewidth=3, linestyle='--', label='Percent Error Difference Mean (' + str(mean_diff) + '%)')

    plt.legend(handles=[green_patch, red_patch, blue_patch, winner_patch])

    # plt.show()

    if save_path is not None:
        plot_file_name = save_path + '_' + model_1 + '_' + model_2 + '_' + 'percent_error_diff.png'
        plt.savefig(plot_file_name)