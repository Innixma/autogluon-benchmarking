Instructions for running auto-sklearn on pandas Dataframe (only for Linux, does not work on Mac):


Install:  

# First need to install swig package.
On Mac: brew install swig@3  # swig version 4.0 produces segfault with autosklearn. Need version 3.
echo 'export PATH="/usr/local/opt/swig@3/bin:$PATH"' >> ~/.bash_profile

On Linux: sudo apt-get install swig

pip install --upgrade pip
pip install auto-sklearn==0.5.2

# 0.6.0 is current auto-sklearn version as of 1/13/2020; AutoML benchmark instead used auto-sklearn version 0.5.1.  However, version 0.6.0 cannot be used in AutoML benchmark due to incompatible sklearn versions. Thus, we instead benchmark with auto-sklearn version 0.5.2 (there do not appear to be major ML / modeling updates between 0.6.0 and 0.5.2).

Also need to install autogluon for data preprocessing:

pip install xgboost pandas numpy sklearn==0.21.2
pip install autogluon mxnet



Example usage is in: example-autosklearn.py

# Note: auto-sklearn==0.5.2 may not place nice with AutoGluon due to different dependency versions.  In this case, please try using auto-sklearn==0.6.0