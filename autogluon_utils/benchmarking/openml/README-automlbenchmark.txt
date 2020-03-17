The OpenML AutoML Benchmark is run using the following repository which you should clone:
https://github.com/Innixma/automlbenchmark

To run automlbenchmark, you must first create an AMI using EC2 instance on which you have already executed the bash script: benchmarking/baselines/install.sh
which installs all the requisite packages.

Then, you must additionally install AutoGluon's custom automlbenchmark by running on your local machine (Recommended):

autogluon_utils/scripts/automlbenchmark/setup-remote-venv.sh --remote-hostname ec2-XX-XX-XXX-XXX.compute-1.amazonaws.com --setup-automlbenchmark --setup-autoweka

Or by running setup-local-venv.sh on your EC2 instance and copying the AutoMLBenchmark code to the EC2 instance, along with the relevant pip installs detailed in setup-remote-venv.sh

Note: Edits may have to be made to setup-remote-venv.sh to ensure the directory paths are aligned with your system.

Then you should properly configure the variables in autogluon_utils/setup/runners/run_automlbenchmark.py.

Finally, you can run the benchmark with the following command:

python autogluon_utils/setup/runners/run_automlbenchmark.py

Once you have started running the benchmark, you can check current status of a particular instance's job by SSHing into the instance and viewing the log-file: benchmark_output/log_autogluon_automlbenchmark.file (this file will also be available in S3 after the job has completed).

To fetch the results, we use command:

python autogluon_utils/benchmarking/openml/aggregate_openml_results.py

To run any individual framework + dataset run, on the EC2 instance you can do the following command:

python automlbenchmark/runbenchmark.py {FRAMEWORK} {DATASET}

For example:

python automlbenchmark/runbenchmark.py autogluon automl_adult_config_small

A full list of frameworks are available in automlbenchmark/resources/frameworks.yaml
A full list of datasets are available in automlbenchmark/resources/benchmarks/
