import os
import time
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from mfes.evaluate_function.hyperparameter_space_utils import get_benchmark_configspace
from mfes.model.rf_with_instances import RandomForestWithInstances
from mfes.utils.util_funcs import get_types
from mfes.acquisition_function.acquisition import EI
from mfes.optimizer.random_sampling import RandomSampling
from mfes.config_space import convert_configurations_to_array
from mfes.config_space import ConfigurationSpace, sample_configurations

plt.switch_backend('agg')


class TSE(object):
    def __init__(self, n_workers=1, method_id='Default'):
        self.method_name = method_id
        self.file_path = "data/%s.npy" % method_id
        self.runtime_limit = None
        self.time_cost = []
        self.inc = 1.
        self.inc_list = []
        self.perf = []
        self.s_time = time.time()
        self.n_workers = n_workers
        os.makedirs('data/xgb', exist_ok=True)
        from mfes.evaluate_function.eval_covtype import load_covtype
        self.X, self.y = load_covtype()
        self.num_cls = 7
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                stratify=self.y, random_state=1)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x_train, self.y_train,
                                                                                  test_size=0.2, stratify=self.y_train,
                                                                                  random_state=1)
        print('x_train shape:', self.x_train.shape)
        print('x_valid shape:', self.x_valid.shape)

        # Basic settings in TSE.
        self.s_max = self.y_train.shape[0]
        # self.s_max = s_max // 100
        self.s_min = self.s_max // 27
        self.s_mid = self.s_max // 5
        self.num_init = 4
        self.num_iter = 28

        self.K = 5
        self.iter_H = 500
        self.iter_L = 20
        self.num_L_init = 100

    def run_parallel_async(self, pool, func, configs):
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

    def objective_function(self, settings):
        params, s = settings
        # Start the clock to determine the cost of this function evaluation
        start_time = time.time()

        # Shuffle the data and split up the request subset of the training data
        shuffle = np.random.permutation(np.arange(self.s_max))
        train_samples = self.x_train[shuffle[:s]]
        train_labels = self.y_train[shuffle[:s]]

        C = params['C']
        kernel = params['kernel']
        degree = params.get('degree', 3)
        gamma = params['gamma']
        coef0 = params.get('coef0', 0)
        tol = params['tol']

        model = SVC(C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    tol=tol,
                    max_iter=1500,
                    random_state=1,
                    decision_function_shape='ovr')
        model.fit(train_samples, train_labels)

        pred = model.predict(self.x_valid)
        acc = accuracy_score(self.y_valid, pred)
        err = 1 - acc
        print(params, acc, time.time() - start_time)

        c = time.time() - start_time

        return err, c

    def mini_smac(self, learn_delta):
        sample_num_m = self.s_mid
        sample_num_l = self.s_min
        if not learn_delta:
            sample_num_m = self.s_min

        start_time = time.time()
        config_space = get_benchmark_configspace('covtype_svm')
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
        for _ in range(self.num_init):
            init_configs = sample_configurations(config_space, self.num_init)
            for config in init_configs:
                perf_t, _ = self.objective_function((config.get_dictionary(), sample_num_m))
                X.append(config)
                y.append(perf_t)
                if perf_t < inc_y:
                    inc_y = perf_t
                c.append([time.time() - start_time, inc_y])
                if learn_delta:
                    perf_l, _ = self.objective_function((config.get_dictionary(), sample_num_l))
                    y_delta.append(perf_t - perf_l)
                else:
                    y_delta.append(perf_t)

        # BO iterations.
        for _ in range(self.num_iter - self.num_init):
            # Update the surrogate model.
            surrogate.train(convert_configurations_to_array(X), np.array(y, dtype=np.float64))

            # Use EI acq to choose next config.
            incumbent = dict()
            best_index = np.argmin(y)
            incumbent['obj'] = y[best_index]
            incumbent['config'] = X[best_index]
            acquisition_func.update(model=surrogate, eta=incumbent)
            next_config = acq_optimizer.maximize(batch_size=1)[0]
            perf_t, _ = self.objective_function((next_config.get_dictionary(), sample_num_m))
            X.append(next_config)
            y.append(perf_t)
            if perf_t < inc_y:
                inc_y = perf_t
            c.append([time.time() - start_time, inc_y])
            if learn_delta:
                perf_l, _ = self.objective_function((next_config.get_dictionary(), sample_num_l))
                y_delta.append(perf_t - perf_l)
            else:
                y_delta.append(perf_t)
        print('end mini %s' % (time.time() - start_time))
        return [convert_configurations_to_array(X), np.array(y_delta, dtype=np.float64)]

    def run(self, train_base_models=True):
        start_time = time.time()

        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=self.n_workers)
        X, y = [], []
        c = []
        inc = 1.
        X_l, y_l = [], []

        weight = np.array([1 / self.K] * (self.K + 1))
        config_evaluated = []
        config_space = get_benchmark_configspace('covtype_svm')
        # Initialize config L.
        config_L = sample_configurations(config_space, self.num_L_init)

        if train_base_models:
            func_configs = list()
            for iter_t in range(self.K):
                print('Build mid fidelity model', iter_t)
                func_configs.append(True)
            func_configs.append(False)
            training_data = self.run_parallel_async(pool, self.mini_smac, func_configs)
            with open('data/xgb/base_%s_data.pkl' % self.method_name, 'wb') as f:
                pickle.dump(training_data, f)
        else:
            with open('data/xgb/base_tse_data_%d.pkl' % 10, 'rb') as f:
                training_data = pickle.load(f)
            print('Load training data for M evaluations!')

        # Create base models.
        base_models = list()
        config_space = get_benchmark_configspace('covtype_svm')
        types, bounds = get_types(config_space)
        for iter_t in range(self.K + 1):
            config_x, config_y = training_data[iter_t]
            model = RandomForestWithInstances(types=types, bounds=bounds)
            model.train(config_x, config_y)
            base_models.append(model)
        low_fidelity_model = base_models[self.K]
        X_l.extend(training_data[self.K][0].tolist())
        y_l.extend(training_data[self.K][1].tolist())
        print('Base model building finished!')

        # The framework of TSE.
        for iter_t in range(self.iter_H):
            print('Iteration in TSE', iter_t)
            # Sample a batch of configurations according to tse model.
            configs = sample_configurations(config_space, self.iter_L * 10)
            config_arrays = convert_configurations_to_array(configs)
            perfs, _ = low_fidelity_model.predict(config_arrays)
            perfs = perfs[:, 0]
            if len(y) > 3:
                preds = []
                for i in range(self.K):
                    m, _ = base_models[i].predict(config_arrays)
                    preds.append(m[:, 0].tolist())
                preds = np.array(preds).T
                preds = np.mat(np.hstack((preds, np.ones((len(configs), 1)))))
                # Add the delta.
                delta = preds * np.mat(weight.reshape(-1, 1))
                perfs += delta.getA()[:, 0]
            configs_candidate = []
            indexes = np.argsort(perfs)[:self.iter_L]
            for index in indexes:
                configs_candidate.append(configs[index])

            # Evaluate the low-fidelity configurations.
            print('=' * 10 + 'Evaluating the low-fidelity configurations')
            config_params = []
            for config in configs_candidate:
                config_params.append((config.get_dictionary(), self.s_min))

            result_perf = self.run_parallel_async(pool, self.objective_function, config_params)

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
                for i in range(self.K):
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
            perf, _ = self.objective_function((next_config.get_dictionary(), self.s_max))
            X.append(next_config)
            y.append(perf)
            if perf < inc:
                inc = perf
            c.append([time.time() - start_time, inc])
            print('Current inc', inc)

            if len(y) < 3:
                continue
            # Learn the weight in TSE.
            Z = []
            for i in range(self.K):
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
            np.save(self.file_path, np.transpose(np.array(c)))
            plt.plot(np.array(c)[:, 0], np.array(c)[:, 1])
            plt.xlabel('time_elapsed (s)')
            plt.ylabel('validation error')
            plt.savefig("data/xgb/%s.png" % self.method_name)
            if time.time() - start_time > self.runtime_limit:
                print('Runtime budget meets!')
                break

        pool.shutdown(wait=True)

    def get_incumbent(self, num_inc=1):
        pass
