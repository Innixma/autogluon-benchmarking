from typing import List
import time

from autogluon_utils.setup.fleet_config.fleet import Fleet
from autogluon_utils.setup.instance_config.instance_config import InstanceConfig
from autogluon_utils.setup.instance_config.instance_benchmark_config import InstanceBenchmarkConfig
from autogluon_utils.setup.automlbenchmark import config_defaults, prepare_instances
from autogluon_utils.setup.instance_config.config_defaults import automlbenchmark_default


# Set session to run true benchmark. Only do this if you know what you are doing.
if __name__ == '__main__':
    fleet_name = 'fleet'
    user = 'ag-user'
    instance_config_default_params = automlbenchmark_default(fleet_name, user)

    datasets, frameworks, yaml_defaults = config_defaults.get_full_benchmark()

    # Use this for a real run, this needs to be run on the instance that will become the AMI to generate config files.
    # instance_benchmark_configs: List[InstanceBenchmarkConfig] = prepare_instances.generate_benchmark_configs(datasets=datasets, frameworks=frameworks, yaml_defaults=yaml_defaults)

    instance_benchmark_configs: List[InstanceBenchmarkConfig] = prepare_instances.get_configs(datasets=datasets, frameworks=frameworks, yaml_defaults=yaml_defaults)

    instance_configs = []
    for config in instance_benchmark_configs:
        instance_config: InstanceConfig = config.construct_instance_config(**instance_config_default_params)
        instance_configs.append(instance_config)

    for config in instance_configs:
        print(config)

    fleet = Fleet(fleet_name=instance_configs[0].fleet_name, instance_config_list=instance_configs, session=None)

    fleet.print_info()
    print('NOTE: Ensure fleet info is accurate, kill the process if not! Sleeping for 30 seconds before creating fleet to give time to abort...')
    time.sleep(30)
    fleet.create_fleet()
    fleet.print_info()
    print('Sleeping for 300 seconds to give time for instances to initialize...')
    time.sleep(300)
    print('Awakening in 5 seconds...')
    time.sleep(5)
    fleet.run_fleet()
    fleet.print_info()
    print('AutoMLBenchmark Benchmark Finished Initialization')
