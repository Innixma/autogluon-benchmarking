#!/bin/bash

set -e

echo $ARGS
echo hello

export OUTPUT_BUCKET='autogluon-tabular'
export OUTPUT_PREFIX='results/automlbenchmark'
export OUTPUT_SUFFIX='core'

source ~/virtual/automlbenchmark/bin/activate
python ~/workspace/automlbenchmark/runbenchmark.py $ARGS

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

echo "saved the results"

exit 0
