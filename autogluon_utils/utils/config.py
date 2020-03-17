from typing import Mapping
import yaml
import tempfile
import os

DEFAULT_CONFIG = {
    's3_path': 'ANONYMOUS',
    'own_team_id': 4289199
}

def config_dir() -> str:
    """
    :return: config directory
    """
    j = os.path.join
    path = j(os.environ.get('XDG_CONFIG_HOME', j(os.environ.get('HOME'), '.config')),  # type: ignore
             'autogluon-utils')
    os.makedirs(path, exist_ok=True)
    return path


def config_file() -> str:
    j = os.path.join
    return j(config_dir(), os.environ.get('AG_UTILS_CONFIG', 'config.yaml'))


def load_config() -> Mapping[str, any]:
    if os.path.exists(config_file()):
        with open(config_file(), 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG
    return config


def save_config(config) -> None:
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        yaml.dump(config, tmpfile)
        os.rename(tmpfile.name, config_file())
