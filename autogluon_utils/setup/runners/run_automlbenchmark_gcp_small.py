from typing import List
import time

from autogluon_utils.setup.fleet_config.fleet import Fleet
from autogluon_utils.setup.instance_config.instance_config import InstanceConfig
from autogluon_utils.setup.instance_config.instance_benchmark_config import InstanceBenchmarkConfig
from autogluon_utils.setup.automlbenchmark import config_defaults, prepare_instances
from autogluon_utils.setup.instance_config.config_defaults import automlbenchmark_default


# Set session to run true benchmark. Only do this if you know what you are doing.
if __name__ == '__main__':
    fleet_name = 'fleet_gcp'
    user = 'ag-user'
    instance_config_default_params = automlbenchmark_default(fleet_name, user)

    # datasets, frameworks, yaml_defaults = config_defaults.get_gcp_small_benchmark()
    datasets, frameworks, yaml_defaults = config_defaults.get_gcp_medium_benchmark()

    # Use this for a real run, this needs to be run on the instance that will become the AMI to generate config files.
    instance_benchmark_configs: List[InstanceBenchmarkConfig] = prepare_instances.generate_benchmark_configs(datasets=datasets, frameworks=frameworks, yaml_defaults=yaml_defaults)

    # instance_benchmark_configs: List[InstanceBenchmarkConfig] = prepare_instances.get_configs(datasets=datasets, frameworks=frameworks, yaml_defaults=yaml_defaults)

    instance_configs = []
    for config in instance_benchmark_configs:
        instance_config: InstanceConfig = config.construct_instance_config(**instance_config_default_params)
        instance_configs.append(instance_config)

    # Small:
    # instance_configs = instance_configs[:5]  # done, 1/5
    # instance_configs = instance_configs[5:10]  # done, 3/5
    # instance_configs = instance_configs[10:15]  # done, 5/5
    # instance_configs = instance_configs[15:20]  # done, 4/5
    # instance_configs = instance_configs[20:25]  # done, 4/5
    # instance_configs = instance_configs[25:30]  # done, 5/5
    # instance_configs = instance_configs[30:35]  # done, 2/5
    # instance_configs = instance_configs[35:]  # done, 4/4

    # Medium:
    # instance_configs = instance_configs[:5]  # started
    # instance_configs = instance_configs[5:10]  # started
    # instance_configs = instance_configs[10:15]  # started
    # instance_configs = instance_configs[15:20]  # started
    # instance_configs = instance_configs[20:25]  # started
    # instance_configs = instance_configs[25:30]  # started
    # instance_configs = instance_configs[30:35]  # started
    instance_configs = instance_configs[35:]  # started

    for i, config in enumerate(instance_configs):
        print(i)
        print(config)

    session = None  # Anonymous
    fleet = Fleet(fleet_name=instance_configs[0].fleet_name, instance_config_list=instance_configs, session=session)

    # for instance in fleet.instance_config_list:
    #     instance.is_created=True
    #     instance.is_running=True

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
    print('AutoMLBenchmark GCP Benchmark Finished Initialization')
