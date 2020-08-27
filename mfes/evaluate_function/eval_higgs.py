from __future__ import division, print_function, absolute_import

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mfes.utils.ease import ease_target


def load_higgs():
    file_path = 'data/higgs/higgs.csv'
    data = pd.read_csv(file_path, delimiter=',', header='infer', na_values=['?'])
    for column in data.columns:
        if any(data[column].isnull()):
            if len(data[column].unique()) < 20:  # Categorical, use mode(0)
                data[column].fillna(value=0, inplace=True)
            else:  # Use mean
                mean_num = data[column].mean()
                data[column].fillna(value=mean_num, inplace=True)
    assert all(data.isnull())
    data = data.values
    x = data[:, 1:]
    y = data[:, 0]
    return x, y


X, y = load_higgs()
num_cls = 2
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('x_train shape:', x_train.shape, x_train.dtype)
print('x_train shape:', x_test.shape, x_test.dtype)
s_max = x_train.shape[0]
resource_unit = s_max // 27


@ease_target(model_dir="./data/models", name='higgs')
def train(resource_num, params, logger=None):
    resource_num = int(resource_num)
    print(resource_num, params)
    global x_train, y_train
    s_max = x_train.shape[0]
    # Create the subset of the full dataset.
    subset_size = resource_num * resource_unit
    shuffle = np.random.permutation(np.arange(s_max))
    train_samples = x_train[shuffle[:subset_size]]
    train_lables = y_train[shuffle[:subset_size]]
    dmtrain = xgb.DMatrix(train_samples, label=train_lables)
    dmvalid = xgb.DMatrix(x_test, label=y_test)

    num_round = 200
    parameters = {}
    for p in params:
        parameters[p] = params[p]

    if num_cls > 2:
        parameters['num_class'] = num_cls
        parameters['objective'] = 'multi:softmax'
        parameters['eval_metric'] = 'merror'
    elif num_cls == 2:
        parameters['objective'] = 'binary:logistic'
        parameters['eval_metric'] = 'error'

    parameters['tree_method'] = 'hist'
    parameters['booster'] = 'gbtree'
    n_thread = 2
    # if resource_num == 27:
    #     n_thread = 3
    parameters['nthread'] = n_thread
    parameters['silent'] = 1
    watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]

    model = xgb.train(parameters, dmtrain, num_round, watchlist, verbose_eval=0)
    pred = model.predict(dmvalid)
    if num_cls == 2:
        pred = [int(i > 0.5) for i in pred]
    acc = accuracy_score(dmvalid.get_label(), pred)
    print(resource_num, params, acc)
    return {'loss': 1 - acc, 'early_stop': False, 'lc_info': []}
