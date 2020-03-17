from autogluon_utils.benchmarking.evaluation.runners import run_generate_clean_openml_full, run_generate_clean_kaggle_full
from autogluon_utils.benchmarking.evaluation.runners import run_evaluation_openml_core, run_evaluation_openml_core10fold, run_evaluation_openml_orig10fold, run_evaluation_openml_orig_vs_core10fold, run_evaluation_openml_core_1h_vs_4h, run_evaluation_openml_ablation, run_evaluation_openml_accuracy
from autogluon_utils.benchmarking.evaluation.runners import run_evaluation_kaggle
from autogluon_utils.benchmarking.evaluation.runners import run_generate_tex_openml
from autogluon_utils.benchmarking.evaluation.runners import run_generate_dataset_x_framework
from autogluon_utils.benchmarking.evaluation.runners import run_move_tex
from autogluon_utils.benchmarking.evaluation.runners import run_generate_plots_openml


# End-to-End results generation, from raw input to finished graphs
def run():
    print('Starting full run...')

    # Clean raw input to standardized format
    run_generate_clean_openml_full.run()
    run_generate_clean_kaggle_full.run()

    # Evaluate openml and kaggle separately
    run_evaluation_openml_core.run()
    run_evaluation_openml_core10fold.run()
    run_evaluation_openml_orig10fold.run()
    run_evaluation_openml_orig_vs_core10fold.run()
    run_evaluation_openml_core_1h_vs_4h.run()
    run_evaluation_openml_ablation.run()
    run_evaluation_openml_accuracy.run()
    run_evaluation_kaggle.run()

    # TODO: Run 1h, 4h, 8h, ablation -> currently hardcoded to 4h for both

    # TODO: Compare original openml to new openml

    # Generate DataFrames
    run_generate_dataset_x_framework.run()

    # Next: Code to generate graphs/plots/etc.

    # Code to convert to LaTeX
    run_generate_tex_openml.run()

    # Code to move tex files to common directory
    run_move_tex.run()

    # Next: Code to generate graphs/plots/etc.
    run_generate_plots_openml.run()

    print('Full run complete!')


if __name__ == '__main__':
    run()
