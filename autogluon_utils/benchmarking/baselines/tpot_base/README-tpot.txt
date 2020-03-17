Instructions for running TPOT on pandas Dataframe:

Install:  

pip install --upgrade pip
pip install TPOT==0.11.1 
# Current version as of 1/8/2020; AutoML benchmark instead used TPOT version 0.9.6

Also need to install other packages for data preprocessing:

pip install xgboost pandas numpy
pip install autogluon mxnet

pip install psutil # only used to find num_cores for custom setting of TPOT argument

Example usage is in: example_tpot.py


Miscellaneous scratch notes:

First load TPOT virtualenv: 

source ~/virtual/tpot/bin/activate