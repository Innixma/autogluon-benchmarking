from autogluon_utils.benchmarking.evaluation.runners import run_generate_clean_openml_full, run_generate_clean_kaggle_full


def run():
    run_generate_clean_openml_full.run()
    run_generate_clean_kaggle_full.run()


if __name__ == '__main__':
    run()
