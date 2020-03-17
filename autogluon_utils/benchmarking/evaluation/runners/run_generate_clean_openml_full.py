from autogluon_utils.benchmarking.evaluation.runners import run_generate_clean_openml_core, run_generate_clean_openml_ablation, run_generate_clean_openml_original, run_generate_clean_openml_autopilot


def run():
    run_generate_clean_openml_core.run()
    run_generate_clean_openml_ablation.run()
    run_generate_clean_openml_original.run()
    run_generate_clean_openml_autopilot.run()


if __name__ == '__main__':
    run()
