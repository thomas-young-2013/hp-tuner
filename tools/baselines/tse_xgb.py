import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--worker', type=int, default=4)
args = parser.parse_args()
sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
sys.path.append('/home/daim/thomas/hp-tuner')

from hoist.model.rf_with_instances import RandomForestWithInstances
from hoist.utils.util_funcs import get_types
from hoist.acquisition_function.acquisition import EI
from hoist.optimizer.random_sampling import RandomSampling
from hoist.config_space import convert_configurations_to_array
from hoist.config_space import ConfigurationSpace, sample_configurations
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter


def load_covtype():
    file_path = 'data/covtype/covtype.data'
    data = pd.read_csv(file_path, delimiter=',', header=None).values
    return data[:, :-1], data[:, -1] - 1


X, y = load_covtype()
num_cls = 7
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('x_train shape:', x_train.shape)
print('x_train shape:', x_test.shape)

# Basic settings in TSE.
s_max = y_train.shape[0]
# s_max = s_max // 100
s_min = s_max // 27
s_mid = s_max // 5
num_init = 4
num_iter = 200

# K = 2
K = 5
iter_H = 500
iter_L = 20
# iter_L = 10
num_L_init = 100


def run_parallel_async(pool, func, configs):
    handlers = []
    for item in configs:
        handlers.append(pool.submit(func, item))

    all_completed = False
    while not all_completed:
        all_completed = True
        for trial in handlers:
            if not trial.done():
                all_completed = False
                time.sleep(0.05)
                break

    result_perf = list()
    for trial in handlers:
        assert (trial.done())
        result_perf.append(trial.result())
    return result_perf


def objective_function(settings):
    params, s = settings
    # Start the clock to determine the cost of this function evaluation
    start_time = time.time()

    # Shuffle the data and split up the request subset of the training data
    shuffle = np.random.permutation(np.arange(s_max))
    train_subset = x_train[shuffle[:s]]
    train_targets_subset = y_train[shuffle[:s]]

    dmtrain = xgb.DMatrix(train_subset, label=train_targets_subset)
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
    parameters['nthread'] = 2
    parameters['silent'] = 1
    watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]

    model = xgb.train(parameters, dmtrain, num_round, watchlist, verbose_eval=0)
    pred = model.predict(dmvalid)
    if num_cls == 2:
        pred = [int(i > 0.5) for i in pred]
    err = 1 - accuracy_score(dmvalid.get_label(), pred)

    c = time.time() - start_time

    return err, c


def create_configspace():
    cs = ConfigurationSpace()
    # n_estimators = UniformFloatHyperparameter("n_estimators", 100, 600, default_value=200, q=10)
    eta = UniformFloatHyperparameter("eta", 0.01, 0.9, default_value=0.3, q=0.01)
    min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, default_value=1, q=0.1)
    max_depth = UniformIntegerHyperparameter("max_depth", 1, 12, default_value=6)
    subsample = UniformFloatHyperparameter("subsample", 0.1, 1, default_value=1, q=0.1)
    gamma = UniformFloatHyperparameter("gamma", 0, 10, default_value=0, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, default_value=1., q=0.1)
    alpha = UniformFloatHyperparameter("alpha", 0, 10, default_value=0., q=0.1)
    _lambda = UniformFloatHyperparameter("lambda", 1, 10, default_value=1, q=0.1)

    cs.add_hyperparameters([eta, min_child_weight, max_depth, subsample, gamma,
                            colsample_bytree, alpha, _lambda])
    return cs


def mini_smac(learn_delta):
    sample_num_m = s_mid
    sample_num_l = s_min
    if not learn_delta:
        sample_num_m = s_min

    start_time = time.time()
    config_space = create_configspace()
    types, bounds = get_types(config_space)
    num_hp = len(bounds)
    surrogate = RandomForestWithInstances(types=types, bounds=bounds)
    acquisition_func = EI(model=surrogate)
    acq_optimizer = RandomSampling(acquisition_func, config_space, n_samples=max(500, 50 * num_hp))
    X = []
    y = []
    y_delta = []
    c = []
    inc_y = 1.

    # Initial design.
    for _ in range(num_init):
        init_configs = sample_configurations(config_space, num_init)
        for config in init_configs:
            perf_t, _ = objective_function((config.get_dictionary(), sample_num_m))
            X.append(config)
            y.append(perf_t)
            if perf_t < inc_y:
                inc_y = perf_t
            c.append([time.time()-start_time, inc_y])
            if learn_delta:
                perf_l, _ = objective_function((config.get_dictionary(), sample_num_l))
                y_delta.append(perf_t - perf_l)
            else:
                y_delta.append(perf_t)

    # BO iterations.
    for _ in range(num_iter - num_init):
        # Update the surrogate model.
        surrogate.train(convert_configurations_to_array(X), np.array(y, dtype=np.float64))

        # Use EI acq to choose next config.
        incumbent = dict()
        best_index = np.argmin(y)
        incumbent['obj'] = y[best_index]
        incumbent['config'] = X[best_index]
        acquisition_func.update(model=surrogate, eta=incumbent)
        next_config = acq_optimizer.maximize(batch_size=1)[0]
        perf_t, _ = objective_function((next_config.get_dictionary(), sample_num_m))
        X.append(next_config)
        y.append(perf_t)
        if perf_t < inc_y:
            inc_y = perf_t
        c.append([time.time() - start_time, inc_y])
        if learn_delta:
            perf_l, _ = objective_function((config.get_dictionary(), sample_num_l))
            y_delta.append(perf_t - perf_l)
        else:
            y_delta.append(perf_t)

    return [convert_configurations_to_array(X), np.array(y_delta, dtype=np.float64)]


def tse(run_id, train_base_models=True):
    start_time = time.time()

    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=args.worker)
    X, y = [], []
    c = []
    inc = 1.
    X_l, y_l = [], []

    weight = np.array([1/K]*(K+1))
    config_evaluated = []
    config_space = create_configspace()
    # Initialize config L.
    config_L = sample_configurations(config_space, num_L_init)

    if train_base_models:
        func_configs = list()
        for iter_t in range(K):
            print('Build mid fidelity model', iter_t)
            func_configs.append(True)
        func_configs.append(False)
        training_data = run_parallel_async(pool, mini_smac, func_configs)
        with open('data/xgb/base_tse_data_%d.pkl' % run_id, 'wb') as f:
            pickle.dump(training_data, f)
    else:
        with open('data/xgb/base_tse_data_%d.pkl' % 10, 'rb') as f:
            training_data = pickle.load(f)
        print('Load training data for M evaluations!')

    # Create base models.
    base_models = list()
    config_space = create_configspace()
    types, bounds = get_types(config_space)
    for iter_t in range(K+1):
        config_x, config_y = training_data[iter_t]
        model = RandomForestWithInstances(types=types, bounds=bounds)
        model.train(config_x, config_y)
        base_models.append(model)
    low_fidelity_model = base_models[K]
    X_l.extend(training_data[K][0].tolist())
    y_l.extend(training_data[K][1].tolist())
    print('Base model building finished!')

    # The framework of TSE.
    for iter_t in range(iter_H):
        print('Iteration in TSE', iter_t)
        # Sample a batch of configurations according to tse model.
        configs = sample_configurations(config_space, iter_L * 10)
        config_arrays = convert_configurations_to_array(configs)
        perfs, _ = low_fidelity_model.predict(config_arrays)
        perfs = perfs[:, 0]
        if len(y) > 3:
            preds = []
            for i in range(K):
                m, _ = base_models[i].predict(config_arrays)
                preds.append(m[:, 0].tolist())
            preds = np.array(preds).T
            preds = np.mat(np.hstack((preds, np.ones((len(configs), 1)))))
            # Add the delta.
            delta = preds*np.mat(weight.reshape(-1, 1))
            perfs += delta.getA()[:, 0]
        configs_candidate = []
        indexes = np.argsort(perfs)[:iter_L]
        for index in indexes:
            configs_candidate.append(configs[index])

        # Evaluate the low-fidelity configurations.
        print('='*10 + 'Evaluating the low-fidelity configurations')
        config_params = []
        for config in configs_candidate:
            config_params.append((config.get_dictionary(), s_min))

        result_perf = run_parallel_async(pool, objective_function, config_params)

        for index, item in enumerate(result_perf):
            X_l.append(configs_candidate[index].get_array().tolist())
            y_l.append(item[0])

        print(np.array(X_l).shape, np.array(y_l, dtype=np.float64).shape)
        # Update f_L.
        print('=' * 10 + 'Retrain the f_L')
        low_fidelity_model.train(np.array(X_l), np.array(y_l, dtype=np.float64))
        config_L.extend(configs_candidate)

        configs_input = []
        for config in config_L:
            if config not in config_evaluated:
                configs_input.append(config)

        # Choose the next configuration.
        config_arrays = convert_configurations_to_array(configs_input)
        perfs, _ = low_fidelity_model.predict(config_arrays)
        perfs = perfs[:, 0]
        if len(y) > 3:
            preds = []
            for i in range(K):
                m, _ = base_models[i].predict(config_arrays)
                preds.append(m[:, 0].tolist())
            preds = np.array(preds).T
            preds = np.mat(np.hstack((preds, np.ones((len(configs_input), 1)))))
            # Add the delta.
            delta = preds * np.mat(weight.reshape(-1, 1))
            perfs += delta.getA()[:, 0]
        next_config = configs_input[np.argmin(perfs)]

        # Evaluate this config with a high-fidelity setting.
        print('=' * 10 + 'Evaluate the high-fidelity configuration')
        perf, _ = objective_function((next_config.get_dictionary(), s_max))
        X.append(next_config)
        y.append(perf)
        if perf < inc:
            inc = perf
        c.append([time.time()-start_time, inc])
        print('Current inc', inc)

        if len(y) < 3:
            continue
        # Learn the weight in TSE.
        Z = []
        for i in range(K):
            m, v = base_models[i].predict(convert_configurations_to_array(X))
            Z.append(m[:, 0].tolist())
        Z = np.mat(np.hstack((np.array(Z).T, np.ones((len(y), 1)))))
        f = np.mat(np.array(y).reshape((-1, 1)))
        # Compute the weight.
        try:
            ZtZ_inv = np.linalg.inv(Z.T * Z)
            weight = (ZtZ_inv * Z.T * f)[:, 0]
            print('The weight updated is', weight)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('Singular matrix encountered, and do not update the weight!')
            else:
                raise ValueError('Unexpected error!')

        # Save the result.
        np.save('data/xgb/tse_%d.npy' % run_id, np.array(c))
        plt.plot(np.array(c)[:, 0], np.array(c)[:, 1])
        plt.xlabel('time_elapsed (s)')
        plt.ylabel('validation error')
        plt.savefig("data/xgb/tse_%d.png" % run_id)
        if time.time() - start_time > 21600:
            raise ValueError('Runtime budget meets!')

    pool.shutdown(wait=True)


if __name__ == "__main__":
    # mini_smac(objective_function)
    # tse(1, train_base_models=False)
    # tse(14, train_base_models=False)
    def trial(id):
        try:
            tse(id, train_base_models=False)
        except ValueError as err:
            print(str(err))
    trial(15)
    trial(16)
    trial(17)
