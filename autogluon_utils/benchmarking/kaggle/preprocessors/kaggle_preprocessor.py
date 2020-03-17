import tempfile
import zipfile
from typing import List, Mapping
import os

import boto3
import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon_utils.configs.kaggle.constants import INDEX_COL


class KaggleRawDatasetPreProcessor:

    def preprocess(self, file_to_location: Mapping[str, str], competition_meta: Mapping[str, any]) -> List[DataFrame]:
        """
        Raw datasets preprocessing

        :param competition_meta: competition metadata
        :param file_to_location: mapping between kaggle file name and location of raw data in s3
        :return: two dataframes in this order: train, test
        """
        raise NotImplementedError('')

    def sanity_print_missing_columns(self, df_test: DataFrame, df_train: DataFrame):
        # Sanity check output
        print(f'   Columns missing in train: {[c for c in df_test.columns if c not in df_train.columns]}')
        print(f'   Columns missing in test : {[c for c in df_train.columns if c not in df_test.columns]}')


class StandardPreProcessor(KaggleRawDatasetPreProcessor):

    def __init__(self, train_name: str = 'train.csv', test_name: str = 'test.csv', drop_cols: List[str] = None):
        self.train_name = train_name
        self.test_name = test_name
        self.drop_cols = drop_cols

    def preprocess(self, file_to_location: Mapping[str, str], competition_meta: Mapping[str, any]) -> List[DataFrame]:
        index_col = competition_meta[INDEX_COL]

        print(' - pre-processing training set')
        df_train = pd.read_csv(file_to_location[self.train_name], index_col=index_col, low_memory=False)

        print(' - pre-processing test set')
        df_test = pd.read_csv(file_to_location[self.test_name], index_col=index_col, low_memory=False)

        self.sanity_print_missing_columns(df_test, df_train)

        if self.drop_cols is not None:
            print(f' - dropping columns {self.drop_cols}')
            df_train = df_train.drop(columns=self.drop_cols)
            df_test = df_test.drop(columns=self.drop_cols)

        return df_train, df_test


class LogLabelTransformPreProcessor(StandardPreProcessor): # needed to log-transform targets for RMSLE metric

    def __init__(self, label_column: str, train_name: str = 'train.csv', test_name: str = 'test.csv', drop_cols: List[str] = None):
        super().__init__(train_name, test_name, drop_cols)
        self.label_column = label_column

    def preprocess(self, file_to_location: Mapping[str, str], competition_meta: Mapping[str, any]) -> List[DataFrame]:
        df_train, df_test = super().preprocess(file_to_location, competition_meta)
        log_target_vals = np.log(df_train[self.label_column] + 1)
        df_train[self.label_column] = log_target_vals
        return df_train, df_test


class IeeeFraudDetectionPreProcessor(KaggleRawDatasetPreProcessor):

    def preprocess(self, file_to_location: Mapping[str, str], competition_meta: Mapping[str, any]) -> List[DataFrame]:
        index_col = competition_meta[INDEX_COL]

        print(' - pre-processing training set')
        df_train_identity = pd.read_csv(file_to_location['train_identity.csv'], index_col=index_col, low_memory=False)
        df_train_transaction = pd.read_csv(file_to_location['train_transaction.csv'], index_col=index_col, low_memory=False)
        df_train = df_train_transaction.join(df_train_identity)

        print(' - pre-processing test set')
        df_test_identity = pd.read_csv(file_to_location['test_identity.csv'], index_col=index_col, low_memory=False)
        df_test_transaction = pd.read_csv(file_to_location['test_transaction.csv'], index_col=index_col, low_memory=False)
        df_test = df_test_transaction.join(df_test_identity)

        # Fix different column names in test set
        df_test = df_test.rename(columns={c: f'id_{c[-2:]}' for c in df_test.columns if (c not in df_train.columns) and (c.startswith('id'))})

        self.sanity_print_missing_columns(df_test, df_train)

        drop_cols = ['TransactionDT']
        df_train = df_train.drop(columns=drop_cols)
        df_test = df_test.drop(columns=drop_cols)

        return df_train, df_test

class SberbankPreProcessor(KaggleRawDatasetPreProcessor):
    def __init__(self, train_name: str = 'train.csv', test_name: str = 'test.csv', drop_cols: List[str] = None):
        self.train_name = train_name
        self.test_name = test_name
        self.drop_cols = drop_cols

    def preprocess(self, file_to_location: Mapping[str, str], competition_meta: Mapping[str, any]) -> List[DataFrame]:
        index_col = competition_meta[INDEX_COL]

        print(' - pre-processing training set')
        macro_df = pd.read_csv(file_to_location['macro.csv'], index_col='timestamp')
        df_train = pd.read_csv(file_to_location[self.train_name], index_col=index_col)
        df_train = df_train.join(macro_df, on='timestamp', how='left')

        print(' - pre-processing test set')
        df_test = pd.read_csv(file_to_location[self.test_name], index_col=index_col)
        df_test = df_test.join(macro_df, on='timestamp', how='left')

        self.sanity_print_missing_columns(df_test, df_train)

        if self.drop_cols is not None:
            print(f' - dropping columns {self.drop_cols}')
            df_train = df_train.drop(columns=self.drop_cols)
            df_test = df_test.drop(columns=self.drop_cols)

        return df_train, df_test



class WalmartRecruitingPreProcessor(KaggleRawDatasetPreProcessor):

    def preprocess(self, file_to_location: Mapping[str, str], competition_meta: Mapping[str, any]) -> List[DataFrame]:
        print(' - pre-processing training set')
        df_train = pd.read_csv(file_to_location['train.csv.zip'], low_memory=False)

        print(' - pre-processing test set')
        # Test set is encrypted with password - extracting encrypted archive
        with tempfile.TemporaryDirectory() as tmp_dir:
            s3 = boto3.client('s3')
            test_archive_zip = 'test.csv.zip'
            test_s3_location = file_to_location[test_archive_zip].replace('s3://', '').split('/')
            bucket = test_s3_location[0]
            key = '/'.join(test_s3_location[1:])
            extract_temp_path = '/'.join([tmp_dir, test_archive_zip])
            s3.download_file(bucket, key, extract_temp_path)

            zip_ref = zipfile.ZipFile(extract_temp_path, 'r')
            zip_ref.extractall(tmp_dir, pwd='Work4WalmarT'.encode())
            zip_ref.close()

            df_test = pd.read_csv('/'.join([tmp_dir, 'test.csv']), low_memory=False)

        return df_train, df_test


class NestedUnzipperPreprocessor:
    def __call__(self, files):
        res = []
        for file in files:
            if os.path.splitext(file)[1].lower() == '.zip':
                print(f"Unzip {file}")
                base = os.path.split(file)[0]
                with zipfile.ZipFile(file, 'r') as zf:
                    zf.extractall(base)
                    res.extend([os.path.join(base, f) for f in zf.namelist()])
                os.unlink(file)
        res = [x for x in res if os.path.isfile(x)]
        return res

