#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle dataset fetcher"""

import pprint as pp
import tempfile
from typing import List, Mapping

import boto3
import os
import logging

from autogluon_utils.configs.kaggle.kaggle_competitions import *
from autogluon_utils.configs.kaggle.common import *
from autogluon_utils.kaggle_tools.kaggle_utils import fetch_kaggle_files, kaggle_api
from autogluon_utils.utils import *

PRE_PROCESSED_TRAIN_FILE_NAME = 'train.csv'
PRE_PROCESSED_TEST_FILE_NAME = 'test.csv'
PRE_PROCESSED_PARQUET_TRAIN_FILE_NAME = 'train.parquet.gz'
PRE_PROCESSED_PARQUET_TEST_FILE_NAME = 'test.parquet.gz'

log = logging.getLogger(__name__)

def fetch(tmpdir: str = None):
    for competition_meta in KAGGLE_COMPETITIONS:
        if not s3_exists_path_prefix(competition_s3_path(competition_meta[NAME])):
            fetch_competition(competition_meta, tmpdir)
        else:
            print(f'Skipping {competition_meta[NAME]}, already in S3')


def fetch_competition(competition_meta: Mapping[str, any], tmpdir: str=None) -> None:
    print(f'================================================================================')
    print(f'Fetching datasets for competition: {competition_meta[NAME]}')
    print(f'================================================================================')
    pp.pprint(competition_meta)

    # Download datasets locally and upload to S3
    s3_path = competition_s3_path(competition_meta[NAME])
    raw_file_to_s3_mapping = fetch_and_upload_to_s3(competition_meta, s3_path, temp_location=tmpdir)

    # Preprocess dataset and upload processed, fit-ready dataset into s3
    preprocess_raw_datasets(s3_path, competition_meta, raw_file_to_s3_mapping)


def preprocess_raw_datasets(competition_s3_path: str, competition_meta: Mapping[str, any], file_to_s3_mapping: Mapping[str, str]):
    print('Starting pre-processing of: %s' % competition_meta[NAME])
    df_train, df_test = competition_meta[PRE_PROCESSOR].preprocess(file_to_s3_mapping, competition_meta)

    print('Writing processed outputs...')
    output_prefix = '/'.join(['s3:/', competition_s3_path, 'processed'])

    train_output_path = f'{output_prefix}/{PRE_PROCESSED_PARQUET_TRAIN_FILE_NAME}'
    _save_parquet(df_train, train_output_path)

    test_output_path = f'{output_prefix}/{PRE_PROCESSED_PARQUET_TEST_FILE_NAME}'
    _save_parquet(df_test, test_output_path)

    train_output_path = f'{output_prefix}/{PRE_PROCESSED_TRAIN_FILE_NAME}'
    _save_csv(df_train, competition_meta, train_output_path)

    test_output_path = f'{output_prefix}/{PRE_PROCESSED_TEST_FILE_NAME}'
    _save_csv(df_test, competition_meta, test_output_path)


def _save_parquet(df_train, train_output_path):
    df_train.to_parquet(train_output_path, compression='gzip')
    print(f' -> {train_output_path}')


def _save_csv(df_train, competition_meta, train_output_path):
    df_train.to_csv(train_output_path, index=competition_meta[INDEX_COL] is not None)
    print(f' -> {train_output_path}')


def fetch_and_upload_to_s3(competition_meta: Mapping[str, any], competition_s3_path: str, temp_location=None) -> Mapping[str, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        if temp_location is not None:
            tmp_dir = temp_location
        print(f'Fetching into temp: {tmp_dir}')
        files = fetch_kaggle_files(competition_meta[NAME], tmp_dir + '/')
        print(f'Retrieved files:')
        pp.pprint(files)
        fetch_processor = competition_meta.get(FETCH_PROCESSOR)
        if fetch_processor:
            files = fetch_processor(files)
        competition_raw_data_s3_path = os.path.join(competition_s3_path, 'raw')
        file_to_s3_mapping = copy_files_to_s3(files, competition_raw_data_s3_path)
    return file_to_s3_mapping


def copy_files_to_s3(files: List[str], competition_raw_data_s3_path: str) -> Mapping[str, str]:
    print('Uploading raw files to S3...')
    s3_client = boto3.client('s3')
    file_to_s3_mapping = {}
    for file in files:
        localfile = os.path.split(file)[1]
        bucket, key_prefix = get_bucket_and_key(competition_raw_data_s3_path)
        key = os.path.join(key_prefix, localfile)
        s3_client.upload_file(file, bucket, key)
        print(f'  - {file} -> s3://{bucket}/{key}')
        file_to_s3_mapping[localfile] = f's3://{bucket}/{key}'
    return file_to_s3_mapping




if __name__ == '__main__':
    fetch()
