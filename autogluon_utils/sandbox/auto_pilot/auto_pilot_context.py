import os
import sagemaker
import time
import uuid

import pandas as pd
from botocore.exceptions import ClientError


class AutoPilotContext:
    def __init__(self, session, job_id, s3_bucket, s3_prefix, role_arn):
        self.session = session
        self.sm = session.client('sagemaker')
        self.job_id = job_id
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.role_arn = role_arn
        self._is_prepared = False
        self._is_started = False
        self._is_finished = False
        self.label = None
        self.objective_func = None
        self.time_limit = None
        self.model_name = None
        self.train_columns = None
        self.num_models_trained = None

    def prepare(self, label, time_limit, objective_func=None):
        self.label = label
        self.time_limit = time_limit
        self.objective_func = objective_func
        self._is_prepared = True

    def fit(self, train_data, label, time_limit, objective_func=None):
        if self._is_started:
            raise AssertionError('AutoPilot already started!')
        if not self._is_prepared:
            self.prepare(label=label, time_limit=time_limit, objective_func=objective_func)
        self.train_columns = [column for column in list(train_data.columns) if column != self.label]
        s3_input_path = self.upload_data_to_s3(data=train_data, filename='train_data.csv', s3_suffix='input', keep_header=True)
        self.create_job(s3_suffix='input')

    def predict(self, test_data):
        if not self._is_finished:
            raise AssertionError('AutoPilot must finish training before predicting!')
        filename = 'test_data.csv'
        if self.label in list(test_data.columns):
            test_data = test_data.drop([self.label], axis=1)
        if self.train_columns is not None:
            test_data = test_data[self.train_columns]
        else:
            print('Warning: self.train_columns not set for predict! If test columns are not aligned, this could cause issues!')
        s3_input_path = self.upload_data_to_s3(data=test_data, filename=filename, s3_suffix='test', keep_header=False)
        s3_output_path = s3_input_path.rsplit('/', 1)[0] + 'predictions'
        test_pred = self.batch_inference(s3_input_path=s3_input_path, s3_output_path=s3_output_path, filename=filename)
        return test_pred

    def batch_inference(self, s3_input_path, s3_output_path, filename):
        uuid_str = str(uuid.uuid4().hex)
        # s3_output_path = s3_output_path + '-' + uuid_str
        batch_job_name = self.model_name + '-' + uuid_str
        if len(batch_job_name) > 60:
            batch_job_name = 'job' + batch_job_name[-60:]

        request = \
            {
                "TransformJobName": batch_job_name,
                "ModelName": self.model_name,
                "BatchStrategy": "MultiRecord",
                "TransformOutput": {
                    "S3OutputPath": s3_output_path
                },
                "TransformInput": {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": s3_input_path
                        }
                    },
                    "ContentType": "text/csv",
                    "SplitType": "Line",
                    "CompressionType": "None"
                },
                "TransformResources": {
                    "InstanceType": "ml.m5.2xlarge",
                    "InstanceCount": 1
                }
            }
        time_start = time.time()
        time_last_check = time_start
        print('Creating batch transform job with name: %s' % batch_job_name)
        self.sm.create_transform_job(**request)
        while True:
            # Needless computation to ensure host isn't shutdown due to low CPU usage
            j = 0
            for i in range(1000):
                j += i
            # time.sleep(30)  # TODO: REMOVE
            time_since_last_check = time.time() - time_last_check

            if time_since_last_check > 30:
                response = self.sm.describe_transform_job(TransformJobName=batch_job_name)
                status = response['TransformJobStatus']
                if status == 'Completed':
                    print("Transform job ended with status: " + status)
                    break
                if status == 'Failed':
                    message = response['FailureReason']
                    print('Transform failed with the following error: {}'.format(message))
                    raise Exception('AutoPilot Transform job failed: %s' % message)
                print("Transform job is still in status: " + status)
                time_elapsed = time.time() - time_start
                print('Inference has taken %ss so far...' % round(time_elapsed, 2))
                time_last_check = time.time()
                if time_elapsed > self.time_limit*2:
                    raise AssertionError('AutoPilot Inference Job has taken > 2x the training time_limit')

        time_end = time.time()
        time_elapsed = time_end - time_start
        print('Inference took %ss' % round(time_elapsed, 2))
        test_prediction_s3_path = s3_output_path + '/' + filename + '.out'
        s3_bucket, s3_object_name = test_prediction_s3_path[5:].split('/', 1)
        s3 = self.session.client('s3')
        local_path = 'tmp/autopilot/' + str(self.job_id) + '/output_predictions/test_predictions.csv'
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(s3_bucket, s3_object_name, local_path)
        test_pred = pd.read_csv(local_path, low_memory=False, names=[self.label])
        return test_pred

    def is_finished(self):
        if self._is_started == False:
            return False
        elif self._is_finished == True:
            return True
        else:
            status = self.get_status()
            print(status)
            job_status = status['AutoMLJobStatus']
            print(job_status)
            if job_status == 'Completed':
                self._is_finished = True
                return True

    def get_status(self):
        status = self.sm.describe_auto_ml_job(AutoMLJobName=self.job_id)
        return status

    def wait_until_is_finished(self):
        time_start = time.time()
        time_last_check = time_start
        while self._is_finished is False:
            time_since_last_check = time.time() - time_last_check
            # Needless computation to ensure host isn't shutdown due to low CPU usage
            j = 0
            for i in range(1000):
                j += i
            if time_since_last_check > 60:
                status = self.get_status()
                print(status)
                state = status['AutoMLJobStatus']
                time_elapsed = time.time() - time_start
                if state == 'Completed':
                    self._is_finished = True
                    print('Job is finished training!')
                elif state == 'Failed':
                    if 'FailureReason' not in status.keys():
                        raise AssertionError('AutoPilot failed training due to an internal error')
                    else:
                        raise AssertionError(status['FailureReason'])
                else:
                    print('Waiting for AutoMLJobStatus to be \'Completed\', currently \'%s\', time elapsed: %ss' % (state, time_elapsed))
                    print('Secondary Status: \'%s\'' % status['AutoMLJobSecondaryStatus'])
                time_last_check = time.time()
                if time_elapsed > self.time_limit*3:
                    raise AssertionError('AutoPilot Training Job has taken > 3x the training time_limit')

    def create_model(self):
        if self._is_finished is False:
            raise AssertionError('AutoPilot must be finished training to create a model!')
        model_name = self.job_id + '-best-model'
        best_model = self.get_best_auto_pilot_model()
        try:
            model_arn = self.sm.create_model(Containers=best_model['InferenceContainers'],
                                        ModelName=model_name,
                                        ExecutionRoleArn=self.role_arn)
        except ClientError:
            print('Model already created, name: %s' % model_name)
        self.model_name = model_name

    def get_best_auto_pilot_model(self):
        if self._is_finished is False:
            raise AssertionError('AutoPilot must be finished training to get best model!')
        candidates = self.sm.list_candidates_for_auto_ml_job(AutoMLJobName=self.job_id, SortBy='FinalObjectiveMetricValue')
        candidates = candidates['Candidates']
        self.num_models_trained = len(candidates)
        if len(candidates) == 0:
            raise AssertionError('AutoPilot did not finish training any models')
        return candidates[0]

    def get_predictor(self):
        print('GET PREDICTOR TODO IMPLEMENT')
        # TODO: Return predictor object that can host inference instances

    def upload_data_to_s3(self, data, filename, s3_suffix='input', keep_header=True):
        local_path = './tmp/autopilot/' + filename
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data.to_csv(local_path, index=False, header=keep_header)
        sess = sagemaker.session.Session(self.session)
        sess.upload_data(path=local_path, bucket=self.s3_bucket, key_prefix=self.s3_prefix + '/' + s3_suffix)
        s3_path = 's3://' + self.s3_bucket + '/' + self.s3_prefix + '/' + s3_suffix
        return s3_path

    def create_job(self, s3_suffix):
        if self._is_started:
            raise AssertionError('AutoPilot already started!')
        auto_ml_job_config = self.construct_auto_ml_job_config()
        input_data_config = self.construct_input_data_config(s3_suffix=s3_suffix)
        output_data_config = self.construct_output_data_config()
        print(auto_ml_job_config)
        print(input_data_config)
        print(output_data_config)
        out = self.sm.create_auto_ml_job(
            AutoMLJobConfig=auto_ml_job_config,
            AutoMLJobName=self.job_id,
            InputDataConfig=input_data_config,
            OutputDataConfig=output_data_config,
            RoleArn=self.role_arn
        )
        self._is_started = True
        return out

    def print_candidates(self):
        candidates = self.sm.list_candidates_for_auto_ml_job(AutoMLJobName=self.job_id, SortBy='FinalObjectiveMetricValue')
        candidates = candidates['Candidates']
        index = 1
        print('####################')
        for candidate in candidates:
            print(str(index) + "  " + candidate['CandidateName'] + "  " + str(candidate['FinalAutoMLJobObjectiveMetric']['Value']))
            index += 1

    def construct_auto_ml_job_config(self):
        auto_ml_job_config = {
            'CompletionCriteria': {
                'MaxAutoMLJobRuntimeInSeconds': self.time_limit,
            }
        }
        return auto_ml_job_config

    def construct_input_data_config(self, s3_suffix):
        input_data_config = [{
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://{}/{}/{}'.format(self.s3_bucket, self.s3_prefix, s3_suffix)
                }
            },
            'TargetAttributeName': self.label
        }
        ]
        return input_data_config

    def construct_output_data_config(self):
        output_data_config = {
            'S3OutputPath': 's3://{}/{}/output'.format(self.s3_bucket, self.s3_prefix)
        }
        return output_data_config
