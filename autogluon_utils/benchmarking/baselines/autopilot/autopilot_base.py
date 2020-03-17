import warnings
import time

import pandas as pd

from ..model_base import AbstractBaseline
from autogluon_utils.sandbox.auto_pilot.auto_pilot_context import AutoPilotContext  # TODO: update location


class AutoPilotBaseline(AbstractBaseline):
    """ Methods to run AutoPilot. """
    def __init__(self):
        super().__init__()
        self.classes = None
        self.auto_pilot_context = None
        self.problem_type = None
        self.label_column = None
        self.label_is_str = None

    def fit(self, train_data, label_column, problem_type, session, job_id, s3_bucket, role_arn, eval_metric = None, runtime_sec = 60):
        self.problem_type = problem_type
        objective_func = self.convert_metric(metric=eval_metric)
        self.label_column = label_column
        self.label_is_str = False
        if self.problem_type != 'regression':
            self.classes = list(train_data[label_column].unique())
            for clss in self.classes:
                if type(clss) == str:
                    self.label_is_str = True
        print(self.classes)

        time_start = time.time()
        auto_pilot_context = AutoPilotContext(session=session, job_id=job_id, s3_bucket=s3_bucket, s3_prefix=job_id, role_arn=role_arn)
        # auto_pilot_context.prepare(label=label_column, time_limit=runtime_sec, objective_func=objective_func)  # TODO: REMOVE
        # auto_pilot_context._is_started = True  # TODO: REMOVE
        # auto_pilot_context.train_columns = [column for column in list(train_data.columns) if column != label_column]  # TODO: REMOVE
        auto_pilot_context.fit(train_data=train_data, label=label_column, time_limit=runtime_sec, objective_func=objective_func)
        auto_pilot_context.wait_until_is_finished()
        auto_pilot_context._is_finished = True
        auto_pilot_context.create_model()
        time_end = time.time()
        fit_time = time_end - time_start

        self.auto_pilot_context = auto_pilot_context
        num_models_trained = self.auto_pilot_context.num_models_trained
        num_models_ensemble = 1

        return num_models_trained, num_models_ensemble, fit_time

    def predict(self, test_data, predict_proba = False, pred_class_and_proba = False):
        """ Use pred_class_and_proba to produce both predicted probabilities and predicted classes.
            If this is regression problem, predict_proba and pred_class_and_proba are disregarded.
            Label column should not be present in test_data.
            
            Returns: Tuple (y_pred, y_prob, inference_time) where any element may be None.
            y_prob is a 2D numpy array of predicted probabilities, where each column represents a class. The ith column represents the class found via: self.classes[i].
        """

        time_start = time.time()

        ###### TODO: Remove
        # local_path = 'tmp/autopilot/output_predictions/test_predictions.csv'
        # os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # s3.download_file(s3_bucket, s3_object_name, local_path)
        # y_pred = pd.read_csv(local_path, low_memory=False, names=[self.label_column])
        ###### TODO: Remove

        y_pred = self.auto_pilot_context.predict(test_data=test_data)
        if self.label_is_str:
            y_pred[self.label_column] = [str(val[0]) for val in zip(y_pred[self.label_column])]
        time_end = time.time()
        predict_time = time_end - time_start

        if self.problem_type != 'regression':
            if predict_proba or pred_class_and_proba:
                y_prob = self.get_proba_from_pred(y_pred)
            else:
                y_prob = None
        else:
            y_prob = None

        return y_pred, y_prob, predict_time

    def convert_metric(self, metric):
        # metrics from https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-auto-ml-job.html
        autopilot_metrics = {
            'accuracy': "Accuracy",
            'f1': 'F1',
            'f1_macro': 'F1macro',
            'mean_squared_error': 'MSE',
        }
        if metric in autopilot_metrics:
            return autopilot_metrics[metric]
        else:
            warnings.warn("Unknown metric will not be used by AutoPilot: %s" % metric)
            return None

    def get_proba_from_pred(self, pred):
        proba = pd.get_dummies(pred)
        proba.columns = [column[len(self.label_column) + 1:] for column in proba.columns]
        proba_columns = list(proba.columns)
        for clss in self.classes:
            if clss not in proba_columns:
                proba[clss] = 0
        proba = proba[self.classes]
        if self.problem_type == 'binary':
            proba = proba[self.classes[1]]
            print(pred)
            print(proba)
            return proba
        else:
            proba = proba.values
            print(pred)
            print(proba)
            return proba
