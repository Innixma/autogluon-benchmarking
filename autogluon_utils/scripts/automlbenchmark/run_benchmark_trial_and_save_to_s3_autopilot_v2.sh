#!/bin/bash

set -e

echo hello

export OUTPUT_BUCKET='autogluon-tabular'
export OUTPUT_PREFIX='results/automlbenchmark'
export OUTPUT_SUFFIX='autopilot'

source ~/virtual/automlbenchmark/bin/activate

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Australian_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_blood-transfusion_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_car_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_cnae-9_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_bank-marketing_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_connect-4_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Fashion-MNIST_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_guiellermo_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Helena_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_higgs_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Jannis_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_jungle_chess_2pcs_raw_endgame_complete_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_KDDCup09_appetency_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_MiniBooNE_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_nomao_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_numerai28-6_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_riccardo_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Robert_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Shuttle_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Volkert_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Airlines_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Albert_config_small_accuracy
python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Covertype_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

python ~/workspace/automlbenchmark/runbenchmark.py AutoPilot automl_Dionis_config_small_accuracy

aws s3 cp results s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive
aws s3 cp logs s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/$OUTPUT_SUFFIX/ --recursive

echo "saved the results"

exit 0
