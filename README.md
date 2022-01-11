# WARNING: THIS CODE IS OUTDATED. IT IS NOT RECOMMENDED TO TRY RUNNING THIS CODE.

This code is from early 2020 and due to dependency version mismatches such as sklearn between the AutoML frameworks in the old versions, it is unlikely to work properly in 2022+.

Instead, please refer to the AutoMLBenchmark repo to run AutoGluon and other AutoML frameworks on OpenML: https://github.com/openml/automlbenchmark

A setup guide for running AutoGluon is available here: https://github.com/Innixma/autogluon-benchmark/blob/master/examples/automlbenchmark/README_automlbenchmark.md

We are working on an updated repo to run the Kaggle competitions, as it unfortunately is quite involved due to a lack of Kaggle API's for fetching leaderboard scores and the lack of other AutoML frameworks compatibility with raw data in Kaggle requiring special preprocessing.

# Original Description

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


