## Install GCP SDK and necessary packages on new Linux machine.

# First make sure GCP key file is available on this machine. 
# If using EC2:  scp -i <AWS_key>.pem <GCP_key>.json <EC2address>:~/

export AUTOGLUON_GIT_URL=<ANONYMOUS_GIT_URL>
export GCP_KEY=<Your_GCP_key>.json # TODO: Place path to GCP key file here.
export GCP_PROJECT=<YOUR_GCP_PROJECT_ID> # TODO: Put your GCP project ID here.

# Install GCP command-line tools:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

source ~/.bashrc # Ensure installation changes take effect

# On personal computer you can interactively activate GCP via command:  gcloud init
# # For non-interactive GCP activation (eg. on EC2):
gcloud auth activate-service-account --project $GCP_PROJECT --key-file $GCP_KEY

# Verify GCP setup using:
gcloud compute project-info describe

# Install Python packages:
pip install --upgrade pip

pip  install numpy pandas mxnet
# may need to do: pip install cython
git clone $AUTOGLUON_GIT_URL
cd autogluon && python setup.py develop

pip install --upgrade google-cloud-automl
pip install --upgrade google-cloud-storage
pip install --upgrade googleapis-common-protos
