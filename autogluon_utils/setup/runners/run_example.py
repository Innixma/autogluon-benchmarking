from typing import List

from autogluon_utils.setup.fleet_config.fleet import Fleet
from autogluon_utils.setup.instance_config.instance_config import InstanceConfig
from autogluon_utils.setup.instance_config.instance_benchmark_config import InstanceBenchmarkConfig


if __name__ == '__main__':
    fleet_name = None  # 'YOUR_FLEET_NAME'
    user = None  # Your alias
    instance_config_default_params = None  # Your config defaults, such as instance type, subnet, etc.

    instance_benchmark_configs: List[InstanceBenchmarkConfig] = None  # Your benchmark configs, such as run_command and name_suffix

    instance_configs = []
    for config in instance_benchmark_configs:
        instance_config: InstanceConfig = config.construct_instance_config(**instance_config_default_params)
        instance_configs.append(instance_config)

    for config in instance_configs:
        print(config)

    session = None  # Your boto3 session
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
    print('Example Benchmark Finished Initialization')
