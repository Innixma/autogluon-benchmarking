Instructions for running auto-WEKA on pandas Dataframe:

# Install WEKA (requires proper JVM version):

bash setup.sh 

# Also need to install autogluon for data preprocessing:

pip install autogluon mxnet psutil

Example usage is in: example_autoweka.py

# In the example, you should set `autoweka_path` variable to the directory from which you ran: bash setup.sh  (this directory should contain lib/autoweka/...)