# Bash script for installation of packages to run autoML baselines.
# Note: all scripts for any baseline method must be run from directory containing the baselines/ folder.
# Note: all scripts for any baseline method must first call: source ~/virtual/baseline/bin/activate


# Before running this script, you must have:

# 1) GCP key file available on this machine: <Your_GCP_key>.json
#
# 2) Kaggle API key. It can be retreived from https://www.kaggle.com/{KAGGLE_USERNAME}/account -> API/Create New API token
# The file must be placed in ~/.kaggle/kaggle.json file.
#
# 3) autogluon-utils must be rsync'd into ~/autogluon-utils

export GCP_KEY=~/YOUR_GCP_KEY.json # TODO: Place path to your GCP key file here.
export GCP_PROJECT=YOUR_GCP_PROJECT_NAME # TODO: Put your GCP project ID here.

export DEBIAN_FRONTEND=noninteractive # disable interactive prompts in apt-get

python --version  # Must be python 3

# Sanity check:
DIRECTORY=~/autogluon-utils/
if [ ! -d "$DIRECTORY" ]; then
    echo "autogluon-utils was not found at $DIRECTORY"
    exit 1
else
    echo "autogluon-utils is found"
fi


# Sanity check: GCP key presence
FILE=$GCP_KEY
if [ ! -f "$FILE" ]; then
    echo "GCP key was not found at $FILE"
    exit 1
else
    echo "GCP key found"
fi

# Sanity check: kaggle key presence
FILE=~/.kaggle/kaggle.json
if [ ! -f "$FILE" ]; then
    echo "Kaggle API key was not found at $FILE"
    exit 1
else
    echo "Kaggle key found"
fi


# sudo apt-get -yq install virtualenv
# python3 -m venv ~/virtual/baseline
# source ~/virtual/baseline/bin/activate # virtual-env is called baseline

pip uninstall --yes TPOT
pip uninstall --yes h2o
pip uninstall --yes auto-sklearn
pip uninstall --yes mxnet
pip uninstall --yes autogluon
pip uninstall --yes scikit-learn
rm -rf ~/autogluon/
rm -rf ~/google-cloud-sdk


pip install --upgrade pip

pip install pyarrow s3fs kaggle fire # requirements of autogluon-utils

pip install xgboost pandas numpy sklearn
pip install requests tabulate future
pip install "colorama>=0.3.8"
pip install psutil # only used to find num_cores for custom setting of TPOT argument

pip install TPOT==0.11.1
pip install h2o==3.28.0.1

sudo apt-get -yq install swig
pip install auto-sklearn==0.5.2

echo "Installed Python AutoML baselines"


pip install pip==19.3.1 # Temporary fix only.
pip install mxnet # If you later get bug during 'import autogluon' with 'cannot import name gluon from mxnet', then need to downgrade pip due to pypi issue:  
git clone <ANONYMOUS_AUTOGLUON_URL>
cd autogluon && python setup.py develop
cd ..
cd autogluon-utils && python setup.py develop
cd ..
pip install --upgrade pip # revert to newest pip

echo "Installed AutoGluon"


# TODO: uninstall sklearn and re-install version  that works with both autogluon and auto-sklearn version 0.5.2.
echo "installing suitable version of scikit-learn..."
pip install scikit-learn==0.20.4

# GCP:
echo "installing GCP..."
curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
# exec -l $SHELL # This may cause early-exit of shell script 
echo "GCP installed, loading gcloud command into path"

source ~/google-cloud-sdk/path.bash.inc # Add gcloud command to path
# source ~/.bashrc # So changes take effect
# source ~/virtual/baseline/bin/activate # Need to reactive virtualenv

echo "activating GCP account..."
# On personal computer for interactive GCP activation: gcloud init 
# For non-interactive GCP activation:
gcloud auth activate-service-account --project $GCP_PROJECT --key-file $GCP_KEY

# Verify install using:
gcloud compute project-info describe

echo "pip installing GCP python packages..."

pip install --upgrade google-cloud-automl
pip install --upgrade google-cloud-storage
pip install --upgrade googleapis-common-protos

echo "Installed GCP"

# Auto-WEKA:
# HERE=$(dirname "$0")
HERE=~
if [[ -x "$(command -v apt-get)" ]]; then
    sudo apt-get update
    sudo apt-get install -y wget unzip openjdk-8-jdk
fi

AUTOWEKA_ARCHIVE="autoweka-2.6.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/autoweka"
if [[ ! -e "$TARGET_DIR" ]]; then
    wget http://www.cs.ubc.ca/labs/beta/Projects/autoweka/$AUTOWEKA_ARCHIVE -P $DOWNLOAD_DIR
    unzip $DOWNLOAD_DIR/$AUTOWEKA_ARCHIVE -d $TARGET_DIR
fi
echo "Installed Auto-WEKA. To run autoweka_fit_predict, you should set autweka_path=$HERE"

# Testing:

echo "Testing Auto-WEKA installation, this should print AutoWEKAClassifier help page..."
java -cp $HERE/lib/autoweka/autoweka.jar weka.classifiers.meta.AutoWEKAClassifier -h

echo "Testing other baselines installation, this should not print anything..."
python -c 'import pandas, numpy, h2o, tpot, autosklearn, autogluon, mxnet, google.cloud'

# Kaggle utils + kaggle API
chmod 600 ~/.kaggle/kaggle.json
pip install --upgrade s3fs
pip install --upgrade pyarrow
pip install --upgrade kaggle
pip install --upgrade fire
python -c 'import s3fs, pyarrow, kaggle, fire'


