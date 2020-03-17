#!/bin/bash

set -e

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Operating System is Mac, running homebrew install of libomp to enable LightGBM"
    # Install libomp to support LightGBM package on mac
    if brew ls --versions libomp > /dev/null; then
        echo "libomp already installed..."
    else
        brew install libomp
    fi
fi

# export MYDIR="$(dirname "$(which "$0")")"
# cd $MYDIR
# cd ../..
# export PROJECT_ROOT_LOCAL=$PWD
export PROJECT_ROOT_LOCAL=~/workspace/autogluon
cd $PROJECT_ROOT_LOCAL

if [[ "$OSTYPE" != "darwin"* ]]; then
    sudo apt-get install -y python3-venv
fi

mkdir -p ~/virtual
python3 -m venv ~/virtual/automlbenchmark
source ~/virtual/automlbenchmark/bin/activate

pip install --upgrade pip
pip install --upgrade mxnet
python setup.py develop
pip install category_encoders  # TODO: Temporary
pip install s3fs  # For benchmarking

echo "AutoGluon install complete"
exit 0
