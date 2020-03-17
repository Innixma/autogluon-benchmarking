""" Variants of AutoGluon for ablation analysis.

Each dict here contains args that should be passed to task.fit(). These args must not be already specified in task.fit!
Example usage:   predictor = task.fit(train_data, label, eval_metric, time_limits, **NO_BAG)

"""

NO_REPEAT_BAG = {
    'auto_stack': True,
    'num_bagging_sets': 1,
}

NO_STACK = {
    'auto_stack': False,
    'num_bagging_folds': 10,
}

NO_BAG = {
    'auto_stack': False,
}

NO_NN = {
    'auto_stack': False,
    'hyperparameters': {
               'GBM': {'num_boost_round': 10000},
               'CAT': {'iterations': 10000},
               'RF': {'n_estimators': 300},
               'XT': {'n_estimators': 300},
               'KNN': {},
               'custom': ['GBM'],
             }
}

NO_KNN = {
    'auto_stack': False,
    'hyperparameters': {
               'NN': {'num_epochs': 500},
               'GBM': {'num_boost_round': 10000},
               'CAT': {'iterations': 10000},
               'RF': {'n_estimators': 300},
               'XT': {'n_estimators': 300},
               'custom': ['GBM'],
             }
}

# Additional ablation is best indvidual model (no ensembling)