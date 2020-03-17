import os, time, logging
import zipfile, glob
import datetime
from typing import List, Mapping
import pandas as pd
from pandas import DataFrame
import tempfile
import bisect
from collections import namedtuple
from retry import retry
from kaggle.rest import ApiException
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.models.kaggle_models_extended import File
from autogluon_utils.utils import *


from autogluon_utils.configs.kaggle.kaggle_competitions import KAGGLE_COMPETITIONS, CONFIG, eval_metric_lower_is_better
from autogluon_utils.configs.kaggle.constants import *
from autogluon_utils.configs.kaggle.common import *

log = logging.getLogger(__name__)


def kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api


class ScoreNotAvailable(RuntimeError):
    pass


"""Info about a submission

public_score: publicScore as reported by kaggle / competition_submissions
private_score: privateScore as reported by kaggle
description: string provided to the submission
leaderboard_result: instance of LeaderBoardResult
error_description: error reported by kaggle api
"""
SubmissionResult = namedtuple('SubmissionResult',
                              ['competition',
                               'public_score',
                               'private_score',
                               'date',
                               'description',
                               'leaderboard_rank',
                               'num_teams',
                               'error_description'])


"""
rank: position in the leaderboard [1, num teams + 1]
"""
LeaderboardRank = namedtuple('LeaderboardRank',
                             ['rank',
                                'num_teams'])


@retry(exceptions=(ApiException), tries=5, delay=10, backoff=5)
def submit_kaggle_competition(competition: str, submission_file: str) -> SubmissionResult:
    """Main wrapper for submitting a competition to kaggle with retries and get scores.
    It will do retries when facing errors, eventually returns SubmissionResult or throws

    :param competition: kaggle competition identifier
    :param submission_file: path to the submission file

    :raises: ScoreNotAvailable, kaggle.rest.ApiException
    :return: instance of SubmissionResult
    :rtype: SubmissionResult
    """
    assert os.path.exists(submission_file)
    api = kaggle_api()
    timestamp = datetime.datetime.utcnow().timestamp()
    description = f'ts: {int(timestamp)}'
    api.competition_submit(submission_file, description, competition)
    # Wait for scoring, competition_score will do exponential retries as well
    time.sleep(30)
    submission_result = competition_score(competition, description)
    if not submission_result.error_description and submission_result.public_score:
        lb_rank = leaderboard_rank(competition, submission_result.public_score)
        submission_result = submission_result._replace(leaderboard_rank=lb_rank.rank, num_teams=lb_rank.num_teams)
    return submission_result


def load_leaderboard_data(competition: str, output_directory: str, force=False) -> DataFrame:
    api = kaggle_api()
    leaderboard_path = output_directory + competition + '/lb/'
    if not os.path.exists(f'{leaderboard_path}/{competition}.zip') or force:
        print(f'Getting Kaggle leaderboard -> {leaderboard_path}')
        api.competition_leaderboard_download(competition, leaderboard_path, quiet=False)
    else:
        print(f'Getting Kaggle leaderboard from local copy -> {leaderboard_path}')

    df = pd.read_csv(f'{leaderboard_path}/{competition}.zip', compression='zip')
    df['__rank__'] = df.groupby('TeamId')['Score'].rank(ascending=True)
    df = df[df['__rank__'] == 1].drop(columns='__rank__').sort_values(by='Score', ascending=True).reset_index(drop=True)
    return df


def fetch_kaggle_files(competition: str, outdir: str, force=False) -> List[File]:
    """Get kaggle files to local and unzip"""
    api = kaggle_api()
    join = os.path.join
    extractdir = join(outdir, competition)
    files_location = join(extractdir, 'files')
    zip_path = join(files_location, f'{competition}.zip')
    if not os.path.exists(zip_path) or force:
        print(f'Getting Kaggle files -> {files_location}')
        api.competition_download_files(competition, path=files_location)
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(extractdir)
        zip_ref.close()
    else:
        print(f'Getting Kaggle files from local copy -> {files_location}')

    files = [os.path.join(extractdir, str(f)) for f in api.competition_list_files(competition)]
    return files


def get_best_submission_scores(competition: str, use_max=True) -> DataFrame:
    api = kaggle_api()
    submissions = pd.DataFrame(api.competitions_submissions_list(competition))
    submissions['publicScore'] = submissions['publicScore'].astype(float)
    best_score = submissions['publicScore'].max() if use_max else submissions['publicScore'].min()
    best_submission = submissions[submissions['publicScore'] == best_score].reset_index(drop=True).iloc[0]
    current_best = pd.DataFrame(best_submission).T[['date', 'privateScore', 'publicScore', 'description']]
    return current_best


def get_percentile(position: int, total_submissions: int) -> float:
    return (total_submissions - position) * 100 / total_submissions


def submit_sample_submissions(datadir: str) -> None:
    """Testing with sample submissions"""
    for competition in KAGGLE_COMPETITIONS:
        print(competition[NAME])
        if os.path.exists(os.path.join(datadir, competition[NAME])):
            log.info("Skipping %s already present", competition[NAME])
            continue
        fd.fetch_kaggle_files(competition[NAME], datadir)
        fetch_processor = competition_meta.get(FETCH_PROCESSOR)
        if fetch_processor:
            files = fetch_processor(files)
    api = kaggle_api()
    for competition in KAGGLE_COMPETITIONS:
        sample_submission = os.path.join(datadir, competition[NAME], 'sample_submission.csv')
        res = submit_kaggle_competition(competition[NAME], sample_submission)
        print(res)


#@retry(exceptions=(ScoreNotAvailable, ApiException), tries=5, delay=10, backoff=5)
def competition_score(competition: str, submission_description: str = None) -> SubmissionResult:
    """returns publicScore, privateScore, date and description as a dictionary
    If submission_description is provided, will return the last one matching that, otherwise just the last one.
    """
    api = kaggle_api()
    submissions = sorted(map(lambda x: x.__dict__, api.competition_submissions(competition)), key=lambda x: x['date'],
                         reverse=True)
    submission_result = None
    if submission_description:
        for sub in submissions:
            if sub['description'] == submission_description:
                submission_result = sub
                break
        if not submission_result:
            raise RuntimeError(f"No submission with description {submission_description}")
    else:
        # last submission
        submission_result = submissions[0]
    x = submission_result
    if x['errorDescription'] is not None:
        public_score = None
        private_score = None
    elif not x['publicScore']:
        raise ScoreNotAvailable(f'Scores are not available: submission result: {str(x)}')
    else:
        public_score = float(x['publicScore'])
        # For competitions in progress the private score will be None
        private_score = float(x['privateScore']) if x['privateScore'] else None

    res = SubmissionResult(competition, public_score, private_score, x['date'], x['description'],
                           None, None, x['errorDescription'])
    return res


def leaderboard_rank(competition: str, public_score: float) -> LeaderboardRank:
    df = pd.read_parquet(competition_lb(competition))
    return _leaderboard_rank(df, public_score, eval_metric_lower_is_better(competition))


def _leaderboard_rank(df: pd.DataFrame, public_score: float, lower_is_better: bool, own_team_id: int=CONFIG['own_team_id']) -> LeaderboardRank:
    """return the leaderboard rank"""
    assert public_score and type(public_score) is float
    log.info("Calculating leaderboard rank")
    df = df.sort_values('Score', ascending=lower_is_better)
    df = df[df['TeamId'] != own_team_id]  # filter own team
    if lower_is_better:
        team_best = df.groupby(['TeamId'])['Score'].min().sort_values()
        rank = team_best.searchsorted(public_score, side='left') + 1
    else:
        team_best = df.groupby(['TeamId'])['Score'].max().sort_values()
        rank = len(team_best) + 1 - team_best.searchsorted(public_score, side='right')
    assert rank >= 1 and rank <= len(team_best) + 1
    return LeaderboardRank(rank, len(team_best))


def load_leaderboard(competition: str) -> pd.DataFrame:
    """return rank in the public leaderboard"""
    api = kaggle_api()
    with tempfile.TemporaryDirectory() as tmpdir, remember_cwd(tmpdir):
        log.info("Downloading leaderboard")
        api.competition_leaderboard_download(competition, tmpdir)
        zf = glob.glob('*.zip')[0]
        df = pd.read_csv(zf, index_col=False)
        return df
