
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd
from autogluon_utils.benchmarking.evaluation import generate_charts


def run():
    results_dir = 'data/results/'
    results_dir_input = results_dir + 'output/'
    results_dir_output = results_dir + 'output/combined/4h/tables/'

    input_openml = results_dir_input + 'openml/core/4h/results_ranked_by_dataset_all.csv'
    input_kaggle = results_dir_input + 'kaggle/4h/results_ranked_by_dataset_all.csv'

    results_ranked_by_dataset_all = load_pd.load([input_openml, input_kaggle])
    print(results_ranked_by_dataset_all)

    result = generate_charts.compute_dataset_framework_df(results_ranked_by_dataset_all)
    print(result)

    save_pd.save(path=results_dir_output + 'dataset_x_framework.csv', df=result)


if __name__ == '__main__':
    run()
