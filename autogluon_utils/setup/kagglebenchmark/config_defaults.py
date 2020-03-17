from ...configs.kaggle.constants import NAME
from ...configs.kaggle.kaggle_competitions import KAGGLE_COMPETITIONS, FITTING_PROFILES, PREDICTORS


def get_full_benchmark(datasets_subset=None, profiles_subset=None, predictors_subset=None):
    """ Can optionally specify just a subset of benchmark to run """
    datasets = [c[NAME] for c in KAGGLE_COMPETITIONS]
    profiles = [p for p in FITTING_PROFILES.keys()]
    predictors = [p for p in PREDICTORS.keys()]
    if datasets_subset is not None:
        datasets = [d for d in datasets if d in datasets_subset]
    if profiles_subset is not None:
        profiles = [p for p in profiles if p in profiles_subset]
    if predictors_subset is not None:
        predictors = [pr for pr in predictors if pr in predictors_subset]
    
    return datasets, profiles, predictors
