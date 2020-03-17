import boto3
import botocore
import contextlib
import os
import re
from typing import List


def get_bucket_and_key(s3_path: str) -> List[str]:
    parts = s3_path.split('/')
    bucket = parts[0]
    key = '/'.join(parts[1:])
    return bucket, key


def s3_exists_file(path: str) -> bool:
    s3 = boto3.resource('s3')
    path = re.sub(r'^s3://', '', path)
    bucket, key = get_bucket_and_key(path)
    try:
        s3.Object(bucket, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise
    return True


def s3_exists_path_prefix(path: str) -> bool:
    s3c = boto3.client('s3')
    paginator = s3c.get_paginator('list_objects')
    bucket, key = get_bucket_and_key(path)
    try:
        pi = paginator.paginate(Bucket=bucket, Prefix=key)
        for x in pi:
            if 'Contents' in x and len(x['Contents']):
                return True
            else:
                break
        return False
    except:
        return False


@contextlib.contextmanager
def remember_cwd(chdir=None):
    '''
    Restore current directory when exiting context
    '''
    curdir = os.getcwd()
    if chdir:
        os.chdir(chdir)
    try: yield
    finally: os.chdir(curdir)


def cache_dir() -> str:
    """
    :return: chache directory
    """
    j = os.path.join
    path = j(os.environ.get('XDG_CACHE_HOME', j(os.environ.get('HOME'), '.cache')),  # type: ignore
             'autogluon-utils')
    os.makedirs(path, exist_ok=True)
    return path

