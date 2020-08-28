from __future__ import division, print_function, absolute_import

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mfes.utils.ease import ease_target


def load_covtype():
    file_path = 'data/covtype/covtype.data'
    data = pd.read_csv(file_path, delimiter=',', header=None).values
    x = data[:, :-1]
    y = data[:, -1] - 1
    return x, y


X, y = load_covtype()
num_cls = 7
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train,
                                                      random_state=1)
print('x_train shape:', x_train.shape)
print('x_train shape:', x_test.shape)
s_max = x_train.shape[0]
resource_unit = s_max // 27


@ease_target(model_dir="./data/models", name='covtype_svm')
def train(resource_num, params, logger=None):
    start_time = time.time()
    resource_num = int(resource_num)
    print(resource_num, params)
    global x_train, y_train
    s_max = x_train.shape[0]
    # Create the subset of the full dataset.
    subset_size = resource_num * resource_unit
    shuffle = np.random.permutation(np.arange(s_max))
    train_samples = x_train[shuffle[:subset_size]]
    train_labels = y_train[shuffle[:subset_size]]

    model = SVC(C=params['C'],
                kernel=params['kernel'],
                degree=params['degree'],
                gamma=params['gamma'],
                coef0=params['coef0'],
                tol=params['tol'],
                max_iter=10000,
                random_state=1,
                decision_function_shape='ovr')
    model.fit(train_samples, train_labels)

    pred = model.predict(x_valid)
    acc = accuracy_score(y_valid, pred)
    print(resource_num, params, acc, time.time() - start_time)
    return {'loss': 1 - acc, 'early_stop': False, 'lc_info': []}
