from ..instance_config.instance_benchmark_config import InstanceBenchmarkConfig

# Path to run_trial script on EC2 instance:
PATH_TO_RUN_TRIAL = '/home/ubuntu/autogluon-utils/autogluon_utils/scripts/kagglebenchmark/run_kaggle_trial.sh'

def get_configs(datasets: list, profiles: list, predictors: list, tag: str = None) -> list:
    base_command = PATH_TO_RUN_TRIAL
    configs = []
    for profile in profiles:
        for predictor in predictors:
            for dataset in datasets:
                name_suffix = '_' + dataset + '_' + profile + '_' + predictor
                run_command = base_command + " " + dataset + " " + profile + " " + predictor
                if tag is not None:
                    run_command = run_command + " " + tag
                config = InstanceBenchmarkConfig(name_suffix=name_suffix, run_command=run_command)
                configs.append(config)
    return configs
