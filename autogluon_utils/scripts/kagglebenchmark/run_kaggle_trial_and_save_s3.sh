#!/bin/bash

set -e

export ARGS="$*"

PATH_TO_EVALUATE_RESULTS="/home/ubuntu/autogluon-utils/autogluon_utils/benchmarking/kaggle/evaluate_results.py"
LOGFILE="python_output.log"
PYTHON_COMMAND="python -u $PATH_TO_EVALUATE_RESULTS $ARGS"
S3LOC="ANONYMOUS"

echo "Now running $0 with args: $ARGS"
echo "Logs will be stored in: $S3LOC"


echo "Running command:"
echo $PYTHON_COMMAND

$PYTHON_COMMAND 2>&1 | tee $LOGFILE

echo "Command completed. Copying output log to s3..."

aws s3 cp $LOGFILE $S3LOC

echo "Finished job. Logs copied to: $S3LOC"

exit 0

