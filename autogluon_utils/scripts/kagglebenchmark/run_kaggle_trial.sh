#!/bin/bash

# Used to run kaggle benchmark on one instance. This script takes same ordered args as evaluate_results: 
# 1 = competition_name, 2 = fitting_profile, 3 = predictor, 4 = tag (not optional!)

set -e

export ARGS="$*"

echo "Now running $0 with: $ARGS"

BENCHMARK_FILE=/home/ubuntu/autogluon-utils/autogluon_utils/scripts/kagglebenchmark/run_kaggle_trial_and_save_s3.sh
COMMAND="$BENCHMARK_FILE $ARGS"

nohup $BENCHMARK_FILE $ARGS > command.out 2>&1 &

echo "Started running command in background:"
echo $COMMAND

exit 0
