autogluon_utils is a package containing code for benchmarking [AutoGluon](https://autogluon.mxnet.io/) and other AutoML frameworks, to accompany the following paper:

[AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505)



To get started, first install this autogluon_utils package using:

**python setup.py develop**


To run the Kaggle benchmark, see instructions in:

**autogluon_utils/benchmarking/kaggle/README-kagglebenchmark.txt**


To run the AutoML benchmark, see instructions in:

**autogluon_utils/benchmarking/openml/README-automlbenchmark.txt**


Code to run the other AutoML frameworks is in the directory:

**autogluon_utils/benchmarking/baselines/**


Files containing our raw benchmark results are available in the **data/results/input/** subdirectory. These results can be formatted as in the paper via code from the directory (**run_full.py** formats all results):

**autogluon_utils/benchmarking/evaluation/runners/**

The formatted results will be stored in: **data/results/output/**


