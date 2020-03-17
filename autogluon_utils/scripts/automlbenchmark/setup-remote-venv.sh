#!/bin/bash

set -e

while test $# -gt 0
do
    case "$1" in
        --setup-automlbenchmark) SETUP_AUTOML_BENCHMARK=true;;
        --setup-autoweka) SETUP_AUTOWEKA=true;;
        --remote-hostname) REMOTE_HOSTNAME="$2";;
    esac
    shift
done

if [ -z "$REMOTE_HOSTNAME" ] ; then
    echo "--remote-hostname is a required parameter (EX: --remote-hostname ec2-XX-XXX-XX-XXX.compute-1.amazonaws.com)"
    exit 1
fi

# Update below if your local paths to the required packages differ
export AUTOGLUON_ROOT_LOCAL=~/workspace/autogluon
export AUTOGLUON_UTILS_ROOT_LOCAL=~/workspace/autogluon-utils/
export AUTOMLBENCHMARK_ROOT_LOCAL=~/workspace/automlbenchmark/
export MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export EXCLUSION_FILE=$MYDIR/exclude_me.txt

# Update below to indicate the EC2 instance you wish to use
# use ec2-user for AL2012 and ubuntu for Ubuntu AMI
export REMOTE_USER=ubuntu
export REMOTE_BOX=$REMOTE_USER@$REMOTE_HOSTNAME

export REMOTE_ROOT=/home/$REMOTE_USER
export WORKSPACE_REMOTE=$REMOTE_ROOT/workspace

export AUTOGLUON_ROOT_REMOTE=$WORKSPACE_REMOTE/autogluon
export AUTOGLUON_UTILS_ROOT_REMOTE=$WORKSPACE_REMOTE/autogluon-utils
export AUTOMLBENCHMARK_ROOT_REMOTE=$WORKSPACE_REMOTE/automlbenchmark
export AUTOMLBENCHMARK_VENV_ROOT_REMOTE=$REMOTE_ROOT/virtual/automlbenchmark

echo $REMOTE_BOX
echo $MYDIR

ssh $REMOTE_BOX mkdir -p $AUTOGLUON_ROOT_REMOTE

rsync --delete -av --exclude-from=$EXCLUSION_FILE $AUTOGLUON_UTILS_ROOT_LOCAL/* $REMOTE_BOX:$AUTOGLUON_UTILS_ROOT_REMOTE/
rsync --delete -av --exclude-from=$EXCLUSION_FILE $AUTOGLUON_ROOT_LOCAL/* $REMOTE_BOX:$AUTOGLUON_ROOT_REMOTE/

ssh $REMOTE_BOX /bin/bash << EOF
cd $AUTOGLUON_UTILS_ROOT_REMOTE/autogluon_utils/scripts/automlbenchmark
./setup-local-venv.sh
EOF

if [ "$SETUP_AUTOML_BENCHMARK" = true ] ; then
rsync --delete -av --exclude-from=$EXCLUSION_FILE $AUTOMLBENCHMARK_ROOT_LOCAL/* $REMOTE_BOX:$AUTOMLBENCHMARK_ROOT_REMOTE/
ssh $REMOTE_BOX /bin/bash << EOF
source $AUTOMLBENCHMARK_VENV_ROOT_REMOTE/bin/activate
pip install -r $AUTOMLBENCHMARK_ROOT_REMOTE/requirements.txt

#pip install requests tabulate future
#pip install "colorama>=0.3.8"
#pip install xgboost
#pip install TPOT==0.11.1
#pip install h2o==3.28.0.1
#
#sudo apt-get -y install swig
#pip install auto-sklearn==0.5.2

pip install scikit-learn==0.20.4
# pip install scipy==1.3.3
EOF

if [ "$SETUP_AUTOWEKA" = true ] ; then
ssh $REMOTE_BOX /bin/bash << EOF
cd $AUTOGLUON_UTILS_ROOT_REMOTE/autogluon_utils/benchmarking/baselines/autoweka/
bash setup.sh
EOF
fi

ssh $REMOTE_BOX /bin/bash << EOF
source $AUTOMLBENCHMARK_VENV_ROOT_REMOTE/bin/activate
cd $AUTOGLUON_UTILS_ROOT_REMOTE
python setup.py develop
EOF
fi

exit 0
