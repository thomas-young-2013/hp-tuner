import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from robo.fmin import fabolas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()


def load_covtype():
    file_path = 'data/covtype/covtype.data'
    data = pd.read_csv(file_path, delimiter=',', header=None).values
    return data[:, :-1], data[:, -1] - 1


s_time = time.time()
X, y = load_covtype()
num_cls = 7
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('x_train shape:', x_train.shape)
print('x_train shape:', x_test.shape)

# metrics
time_cost = []
perf = []
inc = 1.
inc_list = []
run_id = args.run_id
output_path = "data/fabolas_xgb_%d/" % run_id
os.makedirs(output_path, exist_ok=True)


# The optimization function that we want to optimize.
# It gets a numpy array x with shape (D,) where D are the number of parameters
# and s which is the ratio of the training data that is used to
# evaluate this configuration
def objective_function(x, s):
    global inc
    # Start the clock to determine the cost of this function evaluation
    start_time = time.time()

    # Shuffle the data and split up the request subset of the training data
    s_max = y_train.shape[0]
    shuffle = np.random.permutation(np.arange(s_max))
    train_subset = x_train[shuffle[:s]]
    train_targets_subset = y_train[shuffle[:s]]

    dmtrain = xgb.DMatrix(train_subset, label=train_targets_subset)
    dmvalid = xgb.DMatrix(x_test, label=y_test)

    num_round = 200
    parameters = {}
    parameter_names = ['eta', 'min_child_weight', 'max_depth', 'subsample', 'gamma', 'colsample_bytree', 'alpha', 'lambda']
    assert len(parameter_names) == len(x)
    for i, val in enumerate(x):
        if parameter_names[i] == 'max_depth':
            val = int(val)
        parameters[parameter_names[i]] = val

    if num_cls > 2:
        parameters['num_class'] = num_cls
        parameters['objective'] = 'multi:softmax'
        parameters['eval_metric'] = 'merror'
    elif num_cls == 2:
        parameters['objective'] = 'binary:logistic'
        parameters['eval_metric'] = 'error'

    parameters['tree_method'] = 'hist'
    parameters['booster'] = 'gbtree'
    parameters['nthread'] = 2
    parameters['silent'] = 1
    watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]

    model = xgb.train(parameters, dmtrain, num_round, watchlist, verbose_eval=0)
    pred = model.predict(dmvalid)
    if num_cls == 2:
        pred = [int(i > 0.5) for i in pred]
    err = 1 - accuracy_score(dmvalid.get_label(), pred)

    c = time.time() - start_time

    # user-defined operation.
    time_cost.append(time.time() - s_time)
    perf.append(err)
    np.save(output_path + 'int_expdata_xgb.npy', np.array([time_cost, perf]))
    if err < inc:
        inc = err
    inc_list.append(inc)
    np.save(output_path + 'inc_expdata_xgb.npy', np.array([time_cost, inc_list]))

    plt.plot(time_cost, inc_list)
    plt.xlabel('time_elapsed (s)')
    plt.ylabel('validation error')
    plt.savefig(output_path + "fabolas_inc.png")

    if time.time() - s_time > 21600:
        raise ValueError('Runtime budget meets!')
    return err, c


# We optimize s on a log scale, as we expect that the performance varies
# logarithmically across s
s_max = x_train.shape[0]
s_min = s_max // 27
subsets = [27] * 8
subsets.extend([9] * 4)
subsets.extend([3] * 2)
subsets.extend([1] * 1)
# subsets = [64, 32, 16]

# Defining the bounds and dimensions of the
# input space (configuration space + environment space)
# We also optimize the hyperparameters of the svm on a log scale
lower = np.array([0.01, 0, 1, 0.1, 0, 0.1, 0, 1])
upper = np.array([0.9, 10, 12, 1, 10, 1, 10, 10])

# Start Fabolas to optimize the objective function
results = fabolas(objective_function=objective_function, lower=lower, upper=upper,
                  s_min=s_min, s_max=s_max, n_init=len(subsets), num_iterations=1000,
                  n_hypers=30, subsets=subsets, output_path=output_path, inc_estimation="last_seen")

exp_data = np.array([results['runtime'], results['y'].tolist()])
np.save(output_path + 'exp_data_%d.npy', exp_data)
