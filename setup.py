from setuptools import setup, find_packages

requirements = [
    'sagemaker',
    'kaggle',
    'fire',
    's3fs',
    'pyarrow',
    'pandas',
    'numpy',
    'pyyaml',
    'retry',
    'autogluon',
    'mxnet',
    'h2o',
    'tpot',
    'pytz',
]

setup(
    name='autogluon-utils',
    version='0.0.1',
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    url='ANONYMOUS',
    license='ANONYMOUS',
    author='ANONYMOUS',
    install_requires=requirements,
    scripts=['autogluon_utils/bin/agutils'],
    author_email='ANONYMOUS',
    description='utilities for autogluon'
)
