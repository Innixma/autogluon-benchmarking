from typing import Mapping, List

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.utils.tabular.ml.constants import MULTICLASS, BINARY, REGRESSION
from autogluon.utils.tabular.data.label_cleaner import LabelCleanerBinary

from autogluon_utils.configs.kaggle.constants import *


class KaggleSubmissionPostProcessor:
    def postprocess(self, class_labels: List, test_data: DataFrame, y_predproba: np.ndarray, competition_meta: Mapping[str, any]) -> DataFrame:
        raise NotImplementedError('')


class StandardPostProcessor(KaggleSubmissionPostProcessor):
    def postprocess(self, class_labels: List, test_data: DataFrame, y_predproba: np.ndarray, competition_meta: Mapping[str, any]) -> DataFrame:
        if (competition_meta[PROBLEM_TYPE] == BINARY) and (len(y_predproba.shape) > 1) and (y_predproba.shape[1] > 1): 
            # Extract the single correct column of probabilities corresponding to positive class.
            lc = LabelCleanerBinary(pd.Series(sorted(class_labels)))
            positive_class = lc.cat_mappings_dependent_var[1] # this is what autogluon would've chosen as postive class.
            print("Extracting positive class probabilities. Label-column = %s has positive-class = %s" % 
                  (competition_meta[LABEL_COLUMN], positive_class))
            positive_class_index = class_labels.index(positive_class)
            class_labels = positive_class
            y_predproba = y_predproba[:,positive_class_index]
        
        submission = test_data.reset_index()
        index_col = competition_meta[INDEX_COL]
        if competition_meta[PROBLEM_TYPE] == MULTICLASS:
            # multiclass predictions has proba for each column - get those as a separate DataFrame columns
            df_classes = pd.DataFrame(y_predproba)
            df_classes.columns = class_labels
            # Carry-over index from test DataFrame
            df_classes[index_col] = submission[index_col]
            # Keep only index and target classes columns
            submission = df_classes[[index_col, *class_labels]]
        else:
            if LABEL_COLUMN_PRED in competition_meta:
                label_col = competition_meta[LABEL_COLUMN_PRED]
            else:
                label_col = competition_meta[LABEL_COLUMN]
            submission[label_col] = y_predproba
            # Keep only index and predictions
            submission = submission[[index_col, label_col]]
        
        return submission


class LogLabelTransformPostProcessor(StandardPostProcessor): # needed to log-transform targets for RMSLE metric
    def postprocess(self, class_labels: List, test_data: DataFrame, y_predproba: np.ndarray, competition_meta: Mapping[str, any]) -> DataFrame:
        MAXVAL = 1e12
        submission = super().postprocess(class_labels, test_data, y_predproba, competition_meta)
        label_column = competition_meta[LABEL_COLUMN]
        invlog_predictions = np.exp(submission[label_column]) - 1.0
        submission[label_column] = invlog_predictions
        submission[label_column].clip(upper=MAXVAL, inplace=True)
        return submission


class WalmartRecritingPostProcessor(KaggleSubmissionPostProcessor):
    def postprocess(self, class_labels: List, test_data: DataFrame, y_predproba: np.ndarray, competition_meta: Mapping[str, any]) -> DataFrame:
        submission = pd.DataFrame(y_predproba)
        submission.columns = [f'TripType_{c}' for c in class_labels]
        submission = test_data[['VisitNumber']].join(submission)
        submission = submission.groupby('VisitNumber').sum().reset_index()
        return submission

