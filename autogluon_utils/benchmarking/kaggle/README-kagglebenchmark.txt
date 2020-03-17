First, put your kaggle credentials in ~/.kaggle/kaggle.json
You have to accept the rules for all the Kaggle datasets that you download, otherwise you get an error (403).


To run the kaggle benchmark, you must first create AMI using EC2 instance on which you have already executed the bash script: benchmarking/baselines/install.sh
which installs all the requisite packages.

Then you should properly configure the variables in autogluon_utils/setup/runners/run_kaggle.py.

Finally, you can run the benchmark with the following command:

python autogluon_utils/setup/runners/run_kaggle.py 2>&1 | tee outputfile.txt

Command output will be stored in outputfile.txt

Once you have started running the benchmark, you can check current status of a particular instance's job by SSHing into the instance and viewing the log-file: python_output.log (this file will also be available in S3 after the job has completed).


To fetch the results, we use command:

python -u autogluon_utils/reports/my_kaggle_report.py



