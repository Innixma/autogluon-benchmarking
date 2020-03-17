from .utils import get_dataset_list
from ...configs.openml.openml_yaml_small import defaults_yaml_small
from ...configs.openml.openml_yaml_medium import defaults_yaml_medium

AUTOGLUON = 'autogluon'
H2O = 'H2OAutoML_benchmark'
AUTOSKLEARN = 'autosklearn_benchmark'
TPOT = 'TPOT_benchmark'
GCP = 'GoogleAutoMLTables_benchmark'
AUTOPILOT = 'AutoPilot'
AUTOWEKA = 'AutoWEKA_benchmark'


def get_full_benchmark():
    datasets = get_full_datasets()
    frameworks = get_full_frameworks()
    yaml_defaults = get_full_yaml_defaults()
    return datasets, frameworks, yaml_defaults


def get_full_benchmark_ablation():
    datasets = get_full_datasets()
    frameworks = get_ablation_frameworks()
    yaml_defaults = get_full_yaml_defaults()
    return datasets, frameworks, yaml_defaults


def get_full_frameworks():
    return [
        AUTOGLUON,
        H2O,
        AUTOSKLEARN,
        TPOT,
        AUTOWEKA,
    ]


def get_ablation_frameworks():
    return [
        AUTOGLUON + '_nostack',
        AUTOGLUON + '_norepeatbag',
        AUTOGLUON + '_nobag',
        AUTOGLUON + '_nonn',
        AUTOGLUON + '_noknn',
    ]


def get_full_datasets():
    return get_dataset_list()


def get_full_yaml_defaults():
    return [
        defaults_yaml_medium,
        defaults_yaml_small,
        # defaults_yaml_test,
    ]


def get_gcp_small_benchmark():
    datasets = get_full_datasets()
    frameworks = [GCP]
    yaml_defaults = [defaults_yaml_small]
    return datasets, frameworks, yaml_defaults


def get_gcp_medium_benchmark():
    datasets = get_full_datasets()
    frameworks = [GCP]
    yaml_defaults = [defaults_yaml_medium]
    return datasets, frameworks, yaml_defaults


def get_autopilot_benchmark():
    datasets = get_full_datasets()
    frameworks = [AUTOPILOT]
    yaml_defaults = [
        defaults_yaml_small,
        # defaults_yaml_medium
                     ]
    return datasets, frameworks, yaml_defaults
