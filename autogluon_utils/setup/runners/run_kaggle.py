import time
from typing import List

from autogluon_utils.setup.fleet_config.fleet import Fleet
from autogluon_utils.setup.instance_config.instance_config import InstanceConfig
from autogluon_utils.setup.instance_config.instance_benchmark_config import InstanceBenchmarkConfig
from autogluon_utils.setup.instance_config.config_defaults import kagglebenchmark_default
from autogluon_utils.setup.kagglebenchmark import config_defaults, prepare_instances


# Set session to run true benchmark. Only do this if you know what you are doing.
# You can also change which datasets / autoML predictors are run below.
if __name__ == '__main__':
    # Args to change each run: 
    fleet_name = 'fltAmi5-8x-1sthalf-GCP-1h'  # 'YOUR_FLEET_NAME'
    user = 'autogluon-dev'  # Your alias
    
    # Full benchmark:  datasets, profiles, predictors = config_defaults.get_full_benchmark()
    # datasets_subset = None # to run all datasets
    datasets_subset = ['porto-seguro-safe-driver-prediction','santander-customer-satisfaction','santander-customer-transaction-prediction', 'ieee-fraud-detection', 'microsoft-malware-prediction'] # 'otto-group-product-classification-challenge', 'porto-seguro-safe-driver-prediction', 'santander-value-prediction-challenge'] # 'allstate-claims-severity' ]
    
    predictors_subset = ['gcp_predictor'] # 'autogluon_predictor' , 'h2o_predictor', 'tpot_predictor', 'autosklearn_predictor', 'autoweka_predictor']   # Run separately: ['gcp_predictor']
    
    profiles_subset = [
                        # 'PROFILE_FAST',
                        'PROFILE_FULL_1HR',
                        # 'PROFILE_FULL_4HR',
                        # 'PROFILE_FULL_8HR',
                      ]
    datasets, profiles, predictors = config_defaults.get_full_benchmark(datasets_subset=datasets_subset,
                                     predictors_subset=predictors_subset,profiles_subset=profiles_subset)
    print("Running Kaggle benchmark on:")
    print("datasets: ", datasets)
    print("predictors: ", predictors)
    print("profiles: ", profiles)
    
    session = None  # TODO!!!  Your boto3 session, can either be standard boto3
    
    instance_config_default_params = kagglebenchmark_default(fleet_name, user)  # Your config defaults, such as instance type, subnet, etc.
    
    instance_benchmark_configs: List[InstanceBenchmarkConfig] = prepare_instances.get_configs(datasets=datasets, 
                                            profiles=profiles, predictors=predictors, tag=fleet_name)
    
    # Your benchmark configs, such as run_command and name_suffix
    instance_configs = []
    for config in instance_benchmark_configs:
        instance_config: InstanceConfig = config.construct_instance_config(**instance_config_default_params)
        instance_configs.append(instance_config)
    
    for config in instance_configs:
        print(config)
    
    fleet = Fleet(fleet_name=instance_configs[0].fleet_name, instance_config_list=instance_configs, session=session)
    fleet.print_info()
    fleet.create_fleet()  # creates instances on EC2 for every benchmark
    fleet.print_info()
    print('Sleeping for 300 seconds to give time for instances to initialize...')
    time.sleep(300)
    print('Awakening in 5 seconds...')
    time.sleep(5)
    fleet.run_fleet()  # runs instances on EC2 for every benchmark
    fleet.print_info()
    print('Kaggle Benchmark Finished Initialization')
