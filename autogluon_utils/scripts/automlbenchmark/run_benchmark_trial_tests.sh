#!/bin/bash

set -e

echo $ARGS
echo hello

export OUTPUT_BUCKET='autogluon-tabular'
export OUTPUT_PREFIX='results/automlbenchmark'
export OUTPUT_SUFFIX='test'

source ~/virtual/automlbenchmark/bin/activate
mkdir -p test_output
cd test_output

python ~/workspace/automlbenchmark/runbenchmark.py autogluon automl_blood-transfusion_config_test
python ~/workspace/automlbenchmark/runbenchmark.py autosklearn_benchmark automl_blood-transfusion_config_test
python ~/workspace/automlbenchmark/runbenchmark.py H2OAutoML_benchmark automl_blood-transfusion_config_test
python ~/workspace/automlbenchmark/runbenchmark.py TPOT_benchmark automl_blood-transfusion_config_test

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

echo "saved the results"

exit 0