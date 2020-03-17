from ...configs.kaggle.constants import NAME
from ...configs.kaggle.kaggle_competitions import KAGGLE_COMPETITIONS, FITTING_PROFILES, PREDICTORS


def get_full_benchmark():
    datasets = [c[NAME] for c in KAGGLE_COMPETITIONS]
    profiles = [p for p in FITTING_PROFILES.keys()]
    predictors = [p for p in PREDICTORS.keys()]
    return datasets, profiles, predictors
