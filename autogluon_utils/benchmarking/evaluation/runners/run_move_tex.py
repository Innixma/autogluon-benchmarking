import os
import shutil


# Files copied here should be used in the final paper
def run():
    results_dir = 'data/results/output/'
    results_dir_output = 'data/results/paper/tex/'

    openml_tex_files = [
        'core/1h/tex/pairwise/autogluon_1h.tex',
        'core/4h/tex/pairwise/autogluon_4h.tex',
        'orig10fold/1h/tex/pairwise/autogluon_1h.tex',
        'orig10fold/4h/tex/pairwise/autogluon_4h.tex',
        'core10fold/1h/tex/pairwise/autogluon_1h.tex',
        'core10fold/4h/tex/pairwise/autogluon_4h.tex',
        'accuracy/1h/tex/pairwise/autogluon_1h.tex',
        'accuracy/1h/tex/pairwise/autogluon_1h.tex',
        'ablation/1h/tex/pairwise/autogluon_1h.tex',
        'ablation/4h/tex/pairwise/autogluon_4h.tex',
        'ablation/4h/tex/pairwise/autogluon_4h_mini.tex',
        'ablation/combined/tex/pairwise/autogluon_1h.tex',
        'ablation/combined/tex/pairwise/autogluon_4h.tex',
        'core_1h_vs_4h/tex/pairwise/1h_vs_4h.tex',
        'orig_vs_core10fold/tex/pairwise/new_vs_old.tex',
        'accuracy/1h/tex/openml_alllosses_1h.tex',
    ]

    # combined_tex_files = [
    #     '4h/tables/dataset_x_framework.csv'
    # ]

    kaggle_tex_files = [
        '4h/allpercentiles.tex',
        '8h/allpercentiles.tex',
        '4h/pairwise/autogluon_4h.tex',
        '8h/pairwise/autogluon_8h.tex',
        'allVautogluon_4h/pairwise/autogluon_4h.tex',
        'allVautogluon_8h/pairwise/autogluon_8h.tex',
    ]

    openml_tex_files = ['openml/' + file for file in openml_tex_files]
    kaggle_tex_files = ['kaggle/' + file for file in kaggle_tex_files]
    tex_files = openml_tex_files + kaggle_tex_files

    os.makedirs(results_dir_output, exist_ok=True)
    for file in tex_files:
        name = file.replace('/tex/', '_')
        name = name.replace('/', '_')
        old_path = results_dir + file
        new_path = results_dir_output + name
        shutil.copyfile(old_path, new_path)


if __name__ == '__main__':
    run()
