# from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

class AbstractBaseline:
    """ API that all autoML baseline methods should follow """
    def __init__(self):
        self.model = None # The model object used for prediction
        self.classes = None
    
    def fit(self, train_data, label_column, output_directory, problem_type, 
            eval_metric = None, runtime_sec = 60, random_state=0, num_cores=-1):
        """ Fits baseline to given dataset.
            Args:
                train_data: pandas DataFrame 
                    As loaded by autogluon's load_pd().
                label_column: str
                    Name specifying which column contains the target variable values.
                output_directory: str
                    Where to store trained models / auximilary objects.
                problem_type: str
                    What type of prediction problem this is 
                    (options are BINARY, MULTICLASS, REGRESSION from autogluon.utils.tabular.ml.constants).
                eval_metric: str
                    What evaluation metric will be used, n same format as passed to autogluon.TabularPrediction.fit().
                runtime_sec : int
                    Number of seconds this baseline's fit() should run for.
                random_state: int
                    Random seed to pass to baseline fit().
                num_cores: int
                    Number of threads to pass to baseline fit(); set = -1 to use all available cores.
        """
        pass
    
    def predict(self, test_data, predict_proba = False, pred_class_and_proba = False):
        """ Make predictions with fitted model on test_data.
        Use pred_class_and_proba to produce both predicted probabilities and predicted classes.
        If this is regression problem, predict_proba and pred_class_and_proba are disregarded.
        Label column should not be present in test_data.
        
        Returns: Tuple (y_pred, y_prob, predict_time) where any element may be None.
        y_pred : class predictions
             Is = None if this is a regression task.
        y_prob : predicted class probabilities (N x K array, where N = number of rows in test_data, K = number of classes).
            The ith column represents the class found via: self.classes[i].
        predict_time : total time (in sec) needed to produce predictions for all rows in test_data.
    """
        pass
    
    def convert_metric(self, metric):
        """ Converts given evaluation metric string to a format suitable for the baseline method. 
        
        Args:
            metric : str
                Metric in the AutoGluon format.
        Returns:
            str specifying the equivalent metric to use during baseline method's fit().
            None if no appropriate metric can be found.
        """
        return None
