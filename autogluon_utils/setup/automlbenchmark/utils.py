from autogluon_utils.configs.openml.openml_yaml_small import dataset_yaml_small_dict, defaults_yaml_small
from autogluon_utils.configs.openml.openml_yaml_medium import dataset_yaml_medium_dict, defaults_yaml_medium
from autogluon_utils.configs.openml.openml_yaml_large import dataset_yaml_large_dict


def get_dataset_list():
    datasets = []
    for dataset_dict in [dataset_yaml_small_dict, dataset_yaml_medium_dict, dataset_yaml_large_dict]:
        datasets += list(dataset_dict.keys())

    # return list(dataset_yaml_test_dict.keys())
    return datasets
