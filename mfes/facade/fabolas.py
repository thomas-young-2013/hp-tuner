import time
import os
import logging
import numpy as np
import xgboost as xgb
from robo.fmin import fabolas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)


class FABOLAS(object):
    def __init__(self, method_id='Default'):
        self.output_path = "data/%s/" % method_id
        self.file_path = "data/%s.npy" % method_id
        self.runtime_limit = None
        self.time_cost = []
        self.inc = 1.
        self.inc_list = []
        self.perf = []
        self.s_time = time.time()
        os.makedirs(self.output_path, exist_ok=True)

        from mfes.evaluate_function.eval_xgb import load_covtype
        self.X, self.y = load_covtype()
        self.num_cls = 7
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        print('x_train shape:', self.x_train.shape)
        print('x_train shape:', self.x_test.shape)

    # The optimization function that we want to optimize.
    # It gets a numpy array x with shape (D,) where D are the number of parameters
    # and s which is the ratio of the training data that is used to
    # evaluate this configuration
    def objective_function(self, x, s):
        # Start the clock to determine the cost of this function evaluation
        start_time = time.time()
        s = int(s)
        # Shuffle the data and split up the request subset of the training data
        s_max = self.y_train.shape[0]
        shuffle = np.random.permutation(np.arange(s_max))
        train_subset = self.x_train[shuffle[:s]]
        train_targets_subset = self.y_train[shuffle[:s]]

        dmtrain = xgb.DMatrix(train_subset, label=train_targets_subset)
        dmvalid = xgb.DMatrix(self.x_test, label=self.y_test)

        num_round = 200
        parameters = {}
        parameter_names = ['eta', 'min_child_weight', 'max_depth', 'subsample', 'gamma', 'colsample_bytree', 'alpha',
                           'lambda']
        assert len(parameter_names) == len(x)
        for i, val in enumerate(x):
            if parameter_names[i] == 'max_depth':
                val = int(val)
            parameters[parameter_names[i]] = val

        if self.num_cls > 2:
            parameters['num_class'] = self.num_cls
            parameters['objective'] = 'multi:softmax'
            parameters['eval_metric'] = 'merror'
        elif self.num_cls == 2:
            parameters['objective'] = 'binary:logistic'
            parameters['eval_metric'] = 'error'

        parameters['tree_method'] = 'hist'
        parameters['booster'] = 'gbtree'
        parameters['nthread'] = 2
        parameters['silent'] = 1
        watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]

        model = xgb.train(parameters, dmtrain, num_round, watchlist, verbose_eval=0)
        pred = model.predict(dmvalid)
        if self.num_cls == 2:
            pred = [int(i > 0.5) for i in pred]
        err = 1 - accuracy_score(dmvalid.get_label(), pred)

        c = time.time() - start_time

        # user-defined operation.
        self.time_cost.append(time.time() - self.s_time)
        self.perf.append(err)
        np.save(self.output_path + 'int_expdata_xgb.npy', np.array([self.time_cost, self.perf]))
        if err < self.inc:
            self.inc = err
        self.inc_list.append(self.inc)
        np.save(self.file_path, np.array([self.time_cost, self.inc_list]))

        plt.plot(self.time_cost, self.inc_list)
        plt.xlabel('time_elapsed (s)')
        plt.ylabel('validation error')
        plt.savefig(self.output_path + "fabolas_inc.png")

        if time.time() - self.s_time > self.runtime_limit:
            raise ValueError('Runtime budget meets!')
        return err, c

    def run(self):
        # We optimize s on a log scale, as we expect that the performance varies
        # logarithmically across s
        s_max = self.x_train.shape[0]
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
        try:
            results = fabolas(objective_function=self.objective_function, lower=lower, upper=upper,
                              s_min=s_min, s_max=s_max, n_init=len(subsets), num_iterations=1000,
                              n_hypers=30, subsets=subsets, output_path=self.output_path, inc_estimation="last_seen")
        except ValueError as e:
            print(e)

        exp_data = np.array([results['runtime'], results['y'].tolist()])

        np.save(self.output_path + 'exp_data_%d.npy', exp_data)

    def get_incumbent(self, num_inc=1):
        pass
