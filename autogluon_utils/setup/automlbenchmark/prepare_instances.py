from ...configs.openml.openml_yaml_small import dataset_yaml_small_dict
from ...configs.openml.openml_yaml_medium import dataset_yaml_medium_dict
from ...configs.openml.openml_yaml_large import dataset_yaml_large_dict
from ...configs.openml.openml_yaml_test import dataset_yaml_test_dict
from ..instance_config.instance_benchmark_config import InstanceBenchmarkConfig


PATH_TO_AUTOMLBENCHMARK = '../../automlbenchmark/automlbenchmark/'
PATH_TO_AUTOMLBENCHMARK_CONFIGS = PATH_TO_AUTOMLBENCHMARK +  'resources/benchmarks/'

dataset_yaml_full_dict = {}
dataset_yaml_full_dict.update(dataset_yaml_small_dict.copy())
dataset_yaml_full_dict.update(dataset_yaml_medium_dict.copy())
dataset_yaml_full_dict.update(dataset_yaml_large_dict.copy())
dataset_yaml_full_dict.update(dataset_yaml_test_dict.copy())


def get_dataset_list():
    datasets = []
    for dataset_dict in [dataset_yaml_small_dict, dataset_yaml_medium_dict, dataset_yaml_large_dict]:
        datasets += list(dataset_dict.keys())

    # return list(dataset_yaml_test_dict.keys())
    return datasets


def create_yaml_config(dataset, path_to_configs, default_yam):
    dataset_yaml = dataset_yaml_full_dict[dataset]
    name = get_yaml_config_name(dataset, default_yam)
    name_full = path_to_configs + name + '.yaml'
    text_file = open(name_full, 'w')
    n = text_file.write(default_yam[1] + '\n' + dataset_yaml)
    text_file.close()


def create_yaml_config_acc(dataset, path_to_configs, default_yam):
    dataset_yaml = dataset_yaml_full_dict[dataset]

    dataset_yaml = dataset_yaml.split('metric:')[0]
    dataset_yaml = dataset_yaml + 'metric:\n    - acc\n'

    name = get_yaml_config_name(dataset, default_yam)
    name_full = path_to_configs + name + '_accuracy.yaml'
    text_file = open(name_full, 'w')
    n = text_file.write(default_yam[1] + '\n' + dataset_yaml)
    text_file.close()


def get_yaml_config_name(dataset, default_yam, suffix=''):
    # Edge case for numerai28.6, AutoMLBenchmark doesn't like the period and incorrectly saves the results folder to a wrong location
    if '.' in dataset:
        dataset_fixed = dataset.replace('.', '-')
    else:
        dataset_fixed = dataset
    return 'automl_' + dataset_fixed + '_config' + '_' + default_yam[0] + suffix


def get_configs(datasets: list, frameworks: list, yaml_defaults: list, suffix='') -> list:
    configs = []
    for framework in frameworks:
        for default_yaml in yaml_defaults:
            for dataset in datasets:
                name_suffix = '_' + framework + '_' + dataset + '_' + default_yaml[0]
                dataset_yaml_config = get_yaml_config_name(dataset, default_yaml, suffix=suffix)
                run_command = 'workspace/autogluon-utils/autogluon_utils/scripts/automlbenchmark/run_benchmark_trial.sh {framework} {dataset_yaml_config}'.format(dataset_yaml_config=dataset_yaml_config, framework=framework)
                config = InstanceBenchmarkConfig(name_suffix=name_suffix, run_command=run_command)
                configs.append(config)
    return configs


def generate_benchmark_configs(datasets: list, frameworks: list, yaml_defaults: list) -> list:
    # TODO: This won't update AMI
    for default in yaml_defaults:
        for dataset in datasets:
            create_yaml_config(dataset, path_to_configs=PATH_TO_AUTOMLBENCHMARK_CONFIGS, default_yam=default)

    # TODO: Create new AMI here
    configs = get_configs(datasets=datasets, frameworks=frameworks, yaml_defaults=yaml_defaults)

    return configs


def generate_benchmark_configs_accuracy(datasets: list, frameworks: list, yaml_defaults: list) -> list:
    # TODO: This won't update AMI
    for default in yaml_defaults:
        for dataset in datasets:
            create_yaml_config_acc(dataset, path_to_configs=PATH_TO_AUTOMLBENCHMARK_CONFIGS, default_yam=default)

    # TODO: Create new AMI here
    configs = get_configs(datasets=datasets, frameworks=frameworks, yaml_defaults=yaml_defaults, suffix='_accuracy')

    return configs
