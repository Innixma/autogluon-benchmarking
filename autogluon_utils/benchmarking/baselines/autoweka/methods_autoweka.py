""" Methods to run Auto-WEKA """

import pathlib
import time, warnings, os, math, gc
import pandas as pd
import numpy as np
import psutil
from psutil import virtual_memory

from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from autogluon import TabularPrediction as task

from ..process_data import processData
from .csv2arff import Csv2Arff


def autoweka_fit_predict(train_data, test_data, label_column, problem_type, output_directory,
            autoweka_path=None, eval_metric = None, runtime_sec = 60, random_state = 0, num_cores = 1):
    """ Specify different output_directory for each run.
        Args:
            autoweka_path : str
                Folder containing lib/autoweka/autoweka.jar installed during execution 
                of autoweka/setup.sh (must end with / character).
        
        Returns: tuple (num_models_trained, num_models_ensemble, fit_time, y_pred, y_prob, predict_time, class_order)
        where num_models_trained = num_models_ensemble = None,
        class_order indicates the ordering of classes corresponding to columns of y_prob (2D numpy array of predicted probabilties),
        y_pred = pandas Series of predicted classes for test data.
    """
    # if problem_type == REGRESSION:
    #     raise NotImplementedError('Regression is not supported yet')
    if autoweka_path is None:
        working_directory = str(pathlib.Path().absolute())
        autoweka_abs_path = str(pathlib.Path(__file__).parent.absolute())
        autoweka_path = str(os.path.relpath(autoweka_abs_path, working_directory)) + '/'
    # First need to ensure unique labels appear in test data in same order as in training data:
    print("original train_data[label_column]: ", train_data[label_column]) # Replace test labels with dummies
    test_data, dummy_class = dummy_test_labels(train_data, test_data, label_column, problem_type)
    train_data, test_data, class_prefix, labels_are_int = ag_preprocess(train_data, test_data, label_column, 
                                                                        problem_type, eval_metric)
    """ Use this instead of processData if you prefer to run on raw data.  
    # However is very error-prone, eg. will error if there are new feature categories at test-time.
    
    # Weka to requires target as the last attribute: 
    if train_data.columns[-1] != label_column:
        y_train = train_data[label_column]
        train_data.drop([label_column], axis=1, inplace=True)
        train_data[label_column] = y_train
    
    if test_data.columns[-1] != label_column:
        y_test = test_data[label_column]
        test_data.drop([label_column], axis=1, inplace=True)
        test_data[label_column] = y_test
    """
    class_order, train_file_path, test_file_path = data_to_file(train_data, test_data, output_directory, 
                                                                label_column, problem_type)
    fit_predict_time, weka_file, weka_model_file, cmd_root = autoweka_fit(train_file_path=train_file_path, 
            test_file_path=test_file_path, eval_metric=eval_metric, autoweka_path=autoweka_path, 
            output_directory=output_directory, num_cores=num_cores, runtime_sec=runtime_sec, random_state=random_state)
    
    y_pred, y_prob = get_predictions(problem_type=problem_type, weka_file=weka_file, 
                                     class_prefix=class_prefix, labels_are_int=labels_are_int, 
                                     eval_metric=eval_metric)
    fit_time, predict_time = time_predictions(fit_predict_time=fit_predict_time, test_file_path=test_file_path, 
                                              weka_model_file=weka_model_file, cmd_root=cmd_root)
    num_models_ensemble = None
    num_models_trained = None # TODO: hard to get these
    if class_order is not None and len(class_order) > 0:
        if class_order[0].startswith(class_prefix):
            class_order = [clss[len(class_prefix):] for clss in class_order]
        if labels_are_int:
            print("converting classes back to int")
            class_order = [int(clss) for clss in class_order]
    return (num_models_trained, num_models_ensemble, fit_time, y_pred, y_prob, predict_time, class_order)


def dummy_test_labels(train_data, test_data, label_column, problem_type):
    """ Returns copy of test data with dummy test labels for use with auto-weka """
    print("Applying dummy_test_labels...")
    train_label_subset = train_data[label_column].iloc[:len(test_data)].copy()
    dummy_class = train_data[label_column].iloc[0] # placeholder class to use for imputing.
    row = 0
    while pd.isnull(dummy_class):
        row += 1
        if row >= len(train_data):
            raise ValueError("All training labels are missing")
        dummy_class = train_data[label_column].iloc[row].copy()
    
    if len(train_label_subset) < len(test_data):
        num_extra = len(test_data) - len(train_label_subset)
        extra_labels = pd.Series([dummy_class] * num_extra)
        train_label_subset = pd.concat((train_label_subset, extra_labels))
    if len(train_label_subset) != len(test_data):
            raise ValueError("new test labels do not match test-data length")
    if pd.isnull(train_label_subset).any():
        train_label_subset = train_label_subset.fillna(dummy_class)
    
    # train_label_subset.reset_index(drop=True, inplace=True) # otherwise pandas may complain about indexes mismatched
    train_label_subset.index = test_data.index
    if pd.isnull(train_label_subset).any():
        raise ValueError("Error in preprocessing during dummy_test_labels: train_label_subset has missing test labels after index reset")
    
    test_data[label_column] = train_label_subset.copy()
    if pd.isnull(test_data[label_column]).any():
        print(test_data[label_column])
        raise ValueError("Error in preprocessing during dummy_test_labels: cannot have missing test labels")
    
    if problem_type != REGRESSION:
        print(("initial train label uniques: ", set(train_data[label_column])))
        print(("initial test labels uniques: ", set(test_data[label_column])))
        if len(set(test_data[label_column])) > len(set(train_data[label_column])):
            raise ValueError("preprocessing error: somehow failed to replace some test_labels with train_label_subset")
        elif len(set(test_data[label_column])) < len(set(train_data[label_column])):
            # Need to manually go through all training labels and add them to test data:
            unique_classes = set(train_data[label_column])
            num_classes = len(unique_classes)
            class_order = [] # ordering of labels in training data
            remaining_classes = unique_classes.copy()
            i = 0
            train_labels = train_data[label_column].tolist()
            while len(remaining_classes) > 0:
                train_label_i = train_labels[i]
                i += 1
                if train_label_i in remaining_classes:
                    remaining_classes.remove(train_label_i)
                    class_order.append(train_label_i)
                if i > len(test_data):
                    raise ValueError("autoweka preprocessing: Cannot fit all classses into test data")
            for i in range(num_classes): # Note that accuracy of predictions on the first num_class test datapoints will be meaningless.
                test_data.at[i, label_column] = class_order[i]
    
    return (test_data.copy(), dummy_class)


def ag_preprocess(train_data, test_data, label_column, problem_type, eval_metric):
    class_prefix = "cls_"
    if problem_type != REGRESSION:
        print(("after dummy, label uniques: ", set(train_data[label_column])))
        print(("after dummy, label uniques: ", set(test_data[label_column])))
        if len(set(test_data[label_column])) != len(set(train_data[label_column])):
            raise ValueError("preprocessing error: set of test_labels and train_labels must match for auto-weka")
    else:
        print("after dummy, train_data[label_column]: ", train_data[label_column])
        print("after dummy, test_data[label_column]: ", test_data[label_column])
    
    print(train_data.head())
    print(test_data.head())
    
    labels_train = train_data[label_column].copy()
    labels_test = test_data[label_column].copy()
    
    if problem_type == REGRESSION: # ensure labels are numeric
        labels_train = labels_train.astype('float')
        labels_test = labels_test.astype('float')
        test_data[label_column] = labels_test.copy()
        train_data[label_column] = labels_train.copy()
    
    labels_are_int = np.issubdtype(labels_train.dtype, np.signedinteger) # whether true class labels are integers instead of strings
    if pd.isnull(test_data[label_column]).any():
        raise ValueError("Error in data preprocessing, cannot have missing test labels")
    
    print('processing training data')
    X_train, y_train, ag_predictor = processData(data=train_data, label_column=label_column, 
                                                 problem_type=problem_type, eval_metric=eval_metric)
    print('processing test data')
    X_test, y_test, _ = processData(data=test_data, label_column=ag_predictor._learner.label,
                                    ag_predictor=ag_predictor)
    
    if np.sum(X_train.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())) != X_train.shape[1]:
        raise ValueError("AutoGluon-processed train data contains non-numeric values")
    
    if np.sum(X_test.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())) != X_test.shape[1]:
        raise ValueError("AutoGluon-processed test data contains non-numeric values")
    
    if len(X_test) != len(test_data):
        raise ValueError("AutoGluon-processing reduced number of test datapoints. Make sure they have no missing labels!")
    
    # Drop indices: (will error in postprocessing for Kaggle)
    # train_data = X_train.reset_index(drop=True).copy()
    # test_data = X_test.reset_index(drop=True).copy()
    # y_train = y_train.reset_index(drop=True).copy()
    # y_test = y_test.reset_index(drop=True).copy()
    
    # original y_train / y_test from process_data are merely class indices. Turn into original class names:
    y_train = ag_predictor._learner.label_cleaner.inverse_transform(y_train)
    y_test = ag_predictor._learner.label_cleaner.inverse_transform(y_test)
    y_test = labels_test.copy()  # reset test labels to the originals
    if problem_type == BINARY: 
        # special error handling due to issue in AutoGluon Label Cleaner (should be shortly resolved in github)
        y_train_strset = set([str(y) for y in y_train])
        y_test_strset = set([str(y) for y in y_test])
        labels_train_strset = set([str(y) for y in labels_train])
        labels_test_strset = set([str(y) for y in labels_test])
        bool_strset = set([str(True), str(False)])
        int_strset = set([str(0), str(1)])
        if (y_train_strset == bool_strset) and (labels_train_strset == int_strset):
            print("bool int mismatch in train labels")
            y_train = y_train.astype('int')
        elif (y_train_strset == int_strset) and (labels_train_strset == bool_strset):
            print("bool int mismatch in train labels")
            y_train = y_train.astype('bool')
        if (y_test_strset == bool_strset) and (labels_test_strset == int_strset):
            print("bool int mismatch in test labels")
            y_test = y_test.astype('int')
        elif (y_test_strset == int_strset) and (labels_test_strset == bool_strset):
            print("bool int mismatch in test labels")
            y_test = y_test.astype('bool')
    
    if problem_type != REGRESSION:
        print('after autogluon-preprocessing, set(y_train) = ', set(y_train))
        print('after autogluon-preprocessing, set(y_test) = ', set(y_test))
    
    train_data = X_train
    test_data = X_test
    y_train.index = train_data.index
    y_test.index = test_data.index
    
    train_data[label_column] = y_train.copy()
    test_data[label_column] = y_test.copy()
    if problem_type != REGRESSION:
        print(("unique train labels: ", set(train_data[label_column])))
        print(("unique test labels: ", set(test_data[label_column])))
        # Append class prefix to ensure string type used for non-regression problems
        print(len(train_data))
        print(len(y_train))
        print(len(test_data))
        print(len(y_test))
        train_data[label_column] = pd.Series([class_prefix+str(y_train[i]) for i in range(len(train_data))])
        test_data[label_column] = pd.Series([class_prefix+str(y_test[i]) for i in range(len(test_data))])
        print("autoweka train data label_column uniques: ", set(train_data[label_column]))
        print("autoweka test data label_column uniques: ", set(test_data[label_column]))
        print("autoweka train data label_column: ", train_data[label_column])
        print("autoweka test data label_column: ",test_data[label_column])
        if set(train_data[label_column]) != set(test_data[label_column]):
            print(set(train_data[label_column]))
            print(set(test_data[label_column]))
            raise ValueError("preprocessing error: train/test labels differ")
    return (train_data, test_data, class_prefix, labels_are_int)


def data_to_file(train_data, test_data, output_directory, label_column, problem_type):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    train_file_path = output_directory + "train.csv"
    test_file_path = output_directory + "test.csv"
    train_data.to_csv(train_file_path,index=False, header=True) # write processed data to disk
    test_data.to_csv(test_file_path, index=False, header=True)
    # convert to ARFF:
    train_arff_path = output_directory + "train.arff"
    test_arff_path = output_directory + "test.arff"
    converter_args = {'input' : train_file_path, 'output': train_arff_path, 'verbose': True,
                      'label_column': label_column}
    class_order = None
    if problem_type != REGRESSION:
        class_order = sorted(list(set(list(train_data[label_column]))))
        converter_args['class_order'] = class_order
    
    train_converter = Csv2Arff(converter_args)
    converter_args['input'] = test_file_path
    converter_args['output'] = test_arff_path
    test_converter = Csv2Arff(converter_args)
    train_file_path = train_arff_path # Update paths to ARFF files
    test_file_path = test_arff_path
    return (class_order, train_file_path, test_file_path)

def autoweka_metric(eval_metric):
    metrics_mapping = { # Mapping of our metrics to Weka metrics
        'accuracy': 'errorRate',
        'roc_auc': 'areaUnderROC',
        'log_loss': 'kBInformation',
        'mean_squared_error': 'rootMeanSquaredError',
        'mean_absolute_error': 'meanAbsoluteError',
        'r2': 'rootMeanSquaredError',
    }
    metric = metrics_mapping[eval_metric] if eval_metric in metrics_mapping else None
    if metric is None:
        raise ValueError("Performance metric {} not supported.".format(eval_metric))
    return metric

def autoweka_fit(train_file_path, test_file_path, eval_metric, autoweka_path, 
                 output_directory, num_cores, runtime_sec, random_state):
    metric = autoweka_metric(eval_metric)
    training_params = {} # TODO: any non-default auto-WEKA params?
    autoweka_file = autoweka_path + "/lib/autoweka/autoweka.jar"
    preds_file_path = output_directory+"weka_preds.csv"
    weka_file = output_directory + "weka_predNOSLASH.csv" # where predictions are saved 
    weka_model_file = output_directory + "weka.model" # where model is saved
    parallelRuns = num_cores
    if parallelRuns == -1:
         parallelRuns = psutil.cpu_count(logical=False)
    
    mem = virtual_memory()
    total_gb = mem.total >> 30
    memory_limit_mb = (total_gb - 2) * 1000
    memLimit = max(min(memory_limit_mb, math.ceil(memory_limit_mb/float(parallelRuns))), 1024)
    
    cmd_params = dict(
        t='"{}"'.format(train_file_path),
        T='"{}"'.format(test_file_path),
        memLimit=memLimit,
        classifications='"weka.classifiers.evaluation.output.prediction.CSV -distribution -file \"{}\""'.format(weka_file),
        timeLimit=int(runtime_sec/60.0),
        parallelRuns=parallelRuns,
        metric=metric,
        seed=random_state % (1 << 16),   # weka accepts only int16 as seeds
        d=weka_model_file,
        **training_params
    )
    cmd_root = "java -cp {here}lib/autoweka/autoweka.jar weka.classifiers.meta.AutoWEKAClassifier ".format(here=autoweka_path)
    cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in cmd_params.items()])
    
    print("Fitting Auto-WEKA...")
    t0 = time.time()
    errs = os.system(cmd)     # run_cmd(cmd, _live_output_=True)
    print(errs)
    t1 = time.time()
    fit_predict_time = t1 - t0
    print("Finished. Runtime = %s" % fit_predict_time)
    return (fit_predict_time, weka_file, weka_model_file, cmd_root)

def get_predictions(problem_type, weka_file, class_prefix, labels_are_int, eval_metric):
    # Load predictions:
    if not os.path.exists(weka_file):
        raise ValueError("AutoWEKA failed producing any prediction.")
    
    if problem_type in [BINARY, MULTICLASS]: # Load classification predictions:
        # class_labels = sorted(list(set(labels_train)))
        # class_order = [''] * len(class_labels) # will contain ordering of classes
        # remaining_classes = class_labels[:] # classes whose index we don't know yet
        with open(weka_file, 'r') as weka_file_io:
            probabilities = []
            predictions = []
            truth = []
            for line in weka_file_io.readlines()[1:-1]:
                inst, actual, predicted, error, *distribution = line.split(',')
                pred_probabilities = [float(pred_probability.replace('*', '').replace('\n', '')) for pred_probability in distribution]
                _, pred = predicted.split(':')
                _, tru = actual.split(':')
                pred = pred[pred.startswith(class_prefix) and len(class_prefix):]
                if labels_are_int:
                    pred = int(pred)
                probabilities.append(pred_probabilities)
                predictions.append(pred)
                truth.append(tru)
                class_index = np.argmax(pred_probabilities)
                """ # Old code to compute class order:
                if pred in remaining_classes:
                    remaining_classes.remove(pred)
                    class_order[class_index] = pred
                elif class_order[class_index] != pred:
                    raise ValueError("Class ordering cannot be determined due to ordering error")
                """
        """ # Old code to compute class order:
        if len(remaining_classes) > 1:
            raise ValueError("Class ordering cannot be determined because not all classes were predicted")
        elif len(remaining_classes) == 1:
            if '' not in class_order:
                raise ValueError("Class ordering cannot be determined due to error in remaining_classes")
            else:
                remain_idx = class_order.index('')
                class_order[remain_idx] = remaining_classes[0]
        """
        y_pred = pd.Series(predictions)
        y_prob = np.array(probabilities).astype('float')
        if eval_metric == 'log_loss': # ensure there are no probabilities = 0 which may cause infinite loss.
            EPS = 1e-8
            for i in range(len(y_prob)):
                prob_i = y_prob[i]
                extra_prob = 0.0 # additional probability mass.
                for j in range(len(prob_i)):
                    if prob_i[j] == 0.0:
                        prob_i[j] = EPS
                        extra_prob += EPS
                while extra_prob > 0:
                    ind = np.argmax(prob_i)
                    ind_prob = prob_i[ind]
                    if ind_prob > extra_prob:
                        prob_i[ind] = ind_prob - extra_prob
                        extra_prob = 0
                    else:
                        prob_i[ind] = ind_prob - EPS
                        extra_prob -= EPS
        
        y_probsums = np.sum(y_prob, axis=1)
        y_prob = y_prob / y_probsums[:,None] # ensure all probs sum to 1
    elif problem_type == REGRESSION: # Load regression predictions:
        pred_df = load_pd.load(weka_file)
        y_pred = pred_df['predicted']
        y_prob = None
        # class_order = None
    else:
        raise ValueError("Unknown problem_type specified")
    return (y_pred, y_prob)
    
def time_predictions(fit_predict_time, test_file_path, weka_model_file, cmd_root):
    # Call just predict again (just to measure predict-time separately):
    predict_cmd_params = dict(
        T='"{}"'.format(test_file_path),
        l=weka_model_file,
        p=0
    )
    predict_cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in predict_cmd_params.items()])
    print("Using Auto-WEKA for prediction...")
    t2 = time.time()
    errs = os.system(predict_cmd)
    print(errs)
    t3 = time.time()
    predict_time = t3 - t2
    fit_time = fit_predict_time - predict_time
    return (fit_time, predict_time)

"""
## Issues:

- Cannot handle class names with spaces

- 

"""


