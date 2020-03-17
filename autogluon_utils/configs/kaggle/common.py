import os
import posixpath

from .kaggle_competitions import KAGGLE_DATASET_FARM


def competition_s3_path(competition_name: str) -> str:
    return os.path.join(KAGGLE_DATASET_FARM, competition_name)


def competition_lb(competition: str) -> str:
    return posixpath.join('s3://', competition_s3_path(competition), 'leaderboard', f'{competition}_lb.parquet.gz')
