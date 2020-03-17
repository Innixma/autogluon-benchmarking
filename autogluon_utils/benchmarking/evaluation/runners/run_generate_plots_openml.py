import os

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon_utils.benchmarking.evaluation.constants import *


def run():
    time_limit = '1h'
    results_dir = 'data/results/'
    input_subdirectory = 'output/openml/core/'
    results_file = 'results_ranked_by_dataset_all.csv'
    plot_single(results_dir, input_subdirectory, results_file, time_limit)
    time_limit = '4h'
    plot_single(results_dir, input_subdirectory, results_file, time_limit)


def plot_single(results_dir, input_subdirectory, results_file, time_limit):
    print("Producing plots for time = %s" % time_limit)
    input_subdirectory = input_subdirectory+time_limit+'/'
    Rcode = "autogluon_utils/benchmarking/evaluation/plotting/openmlbenchmarkplots.R"
    command = " ".join(["Rscript --vanilla", Rcode, results_dir, input_subdirectory, results_file, time_limit])
    returned_value = os.system(command)
    print("R script exit code: %s" % returned_value)


if __name__ == '__main__':
    run()
