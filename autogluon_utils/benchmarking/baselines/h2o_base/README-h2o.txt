Instructions for running h2o on pandas Dataframe:

Install:  

pip install --upgrade pip
pip install requests
pip install tabulate
pip install "colorama>=0.3.8"
pip install future

pip uninstall h2o

pip install h2o==3.28.0.1


# Current version as of 1/12/2020; AutoML benchmark instead used h2o version 3.24.0.1 


Also need to install autogluon for data preprocessing:

pip install xgboost pandas numpy sklearn psutil
pip install autogluon mxnet



Example usage is in: example_h2o.py