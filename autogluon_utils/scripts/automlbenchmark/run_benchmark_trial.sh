#!/bin/bash

set -e

export ARGS="$*"

echo $ARGS
echo hello

export MYDIR="$(dirname "$(which "$0")")"
mkdir -p benchmark_output
cd benchmark_output

echo $MYDIR

nohup ~/$MYDIR/run_benchmark_trial_and_save_to_s3.sh > log_autogluon_automlbenchmark.file 2>&1 &

echo "initialized code:"
echo "~/$MYDIR/run_benchmark_trial_and_save_to_s3.sh > log_autogluon_automlbenchmark.file 2>&1 &"

exit 0
