import time
import itertools
import numpy as np
from math import log, ceil
from scipy.optimize import minimize
from hoist.utils.util_funcs import get_types
from hoist.facade.base_facade import BaseFacade
from hoist.acquisition_function.acquisition import EI
from hoist.utils.util_funcs import minmax_normalization
from hoist.config_space.util import expand_configurations
from hoist.optimizer.random_sampling import RandomSampling
from hoist.model.rf_with_instances import RandomForestWithInstances
from hoist.model.weighted_rf_ensemble import WeightedRandomForestCluster
from hoist.config_space import convert_configurations_to_array, sample_configurations


class HOIST(BaseFacade):

    def __init__(self, config_space, objective_func, R, num_iter=10, eta=3, n_workers=1):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.config_space = config_space
        self.R = R
        self.eta = eta
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.num_iter = num_iter

        types, bounds = get_types(config_space)
        self.num_config = len(bounds)

        # Define the multi-fidelity ensemble surrogate.
        init_weight = [0.]
        init_weight.extend([1/self.s_max]*self.s_max)
        self.weighted_surrogate = WeightedRandomForestCluster(types, bounds, self.s_max, self.eta, init_weight, 'gpoe')
        self.weighted_acquisition_func = EI(model=self.weighted_surrogate)
        self.weighted_acq_optimizer = RandomSampling(self.weighted_acquisition_func,
                                                     config_space, n_samples=max(1000, 50*self.num_config))

        self.incumbent_configs = []
        self.incumbent_obj = []
        self.iterate_id = 0
        self.iterate_r = []

        # Store the multi-fidelity evaluation data: D_1, ..., D_K.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max+1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = []
            self.target_y[r] = []

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("HOIST algorithm: %d/%d iteration starts." % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.iterate_id += 1
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):
            # Initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # Initial number of iterations per config
            r = int(self.R * self.eta ** (-s))

            # Choose a batch of configurations in different mechanisms.
            start_time = time.time()
            T = self.choose_next(n)
            time_elapsed = time.time() - start_time
            self.logger.info("Choosing next configurations took %.2f sec." % time_elapsed)

            extra_info = None
            last_run_num = None

            for i in range((s + 1) - int(skip_last)):  # Changed from s + 1

                # Run each of the n configs for <iterations> and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                n_iter = n_iterations
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iterations

                self.logger.info("HOIST: %d configurations WITH %d units of resource" % (int(n_configs), int(n_iterations)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]

                self.target_x[int(n_iterations)].extend(T)
                self.target_y[int(n_iterations)].extend(val_losses)

                if int(n_iterations) == self.R:
                    self.incumbent_configs.extend(T)
                    self.incumbent_obj.extend(val_losses)

                # Select a number of well-performed configurations for the next loop.
                indices = np.argsort(val_losses)
                if len(T) == sum(early_stops):
                    break
                if len(T) >= self.eta:
                    T = [T[i] for i in indices if not early_stops[i]]
                    extra_info = [ref_list[i] for i in indices if not early_stops[i]]
                    reduced_num = int(n_configs / self.eta)
                    T = T[0:reduced_num]
                    extra_info = extra_info[0:reduced_num]
                else:
                    T = [T[indices[0]]]
                    extra_info = [ref_list[indices[0]]]
                incumbent_loss = val_losses[indices[0]]
                self.add_stage_history(self.stage_id, min(self.global_incumbent, incumbent_loss))
                self.stage_id += 1
            self.remove_immediate_model()

            # Augment the intermediate evaluation data.
            for item in self.iterate_r[self.iterate_r.index(r):]:
                # objective value normalization: min-max linear normalization
                normalized_y = minmax_normalization(self.target_y[item])
                self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                              np.array(normalized_y, dtype=np.float64), r=item)
            # Update the parameter in the ensemble model.
            if len(self.target_y[self.iterate_r[-1]]) >= 2:
                self.update_weight()

    def choose_next(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            return sample_configurations(self.config_space, num_config)

        conf_cnt = 0
        next_configs = []
        total_cnt = 0

        incumbent = dict()
        max_r = self.iterate_r[-1]
        best_index = np.argmin(self.target_y[max_r])
        incumbent['config'] = self.target_x[max_r][best_index]
        approximate_obj = self.weighted_surrogate.predict(convert_configurations_to_array([incumbent['config']]))[0]
        incumbent['obj'] = approximate_obj
        self.weighted_acquisition_func.update(model=self.weighted_surrogate, eta=incumbent)

        while conf_cnt < num_config and total_cnt < 2 * num_config:
            rand_config = self.weighted_acq_optimizer.maximize(batch_size=1)[0]
            if rand_config not in next_configs:
                next_configs.append(rand_config)
                conf_cnt += 1
            total_cnt += 1
        if conf_cnt < num_config:
            next_configs = expand_configurations(next_configs, self.config_space, num_config)
        return next_configs

    def update_weight(self):
        max_r = self.iterate_r[-1]
        r_list = self.iterate_r

        incumbent_configs = self.target_x[max_r]
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = minmax_normalization(self.target_y[max_r])

        predictions = []
        for i, r in enumerate(r_list[:-1]):
            mean, _ = self.weighted_surrogate.surrogate_container[r].predict(test_x)
            predictions.append(mean.flatten().tolist())
        predictions.append(self.obtain_cv_prediction(test_x, np.array(test_y, dtype=np.float64)))
        solution, status = self.solve_optpro(np.mat(predictions).T, np.mat(test_y).T)
        if status:
            solution[solution < 1e-3] = 0.
            self.logger.info('New weight: %s' % str(solution))
            for i, r in enumerate(r_list):
                self.weighted_surrogate.surrogate_weight[r] = solution[i]

    def obtain_cv_prediction(self, X, y):
        types, bounds = get_types(self.config_space)
        base_model = RandomForestWithInstances(types=types, bounds=bounds)
        instance_num = len(y)
        output_pred = []
        if instance_num < 10:
            for i in range(instance_num):
                row_indexs = list(range(instance_num))
                del row_indexs[i]
                base_model.train(X[row_indexs], y[row_indexs])
                mu, _ = base_model.predict(X)
                output_pred.append(mu[i, 0])
        else:
            # Conduct 5-fold cross validation.
            K = 5
            fold_num = instance_num // K
            for i in range(K):
                row_indexs = list(range(instance_num))
                bound = (instance_num - i * fold_num) if i == (K - 1) else fold_num
                for index in range(bound):
                    del row_indexs[i * fold_num]
                base_model.train(X[row_indexs, :], y[row_indexs])
                mu, _ = base_model.predict(X)
                start = i*fold_num
                end = start + bound
                output_pred.extend(mu[start:end, 0].tolist())
        assert len(output_pred) == instance_num
        return output_pred

    def solve_optpro(self, pred_y, true_y, debug=False):

        # The optimization function.
        def Loss_func(true_y, pred_y):
            # Compute the rank loss for varied loss function.
            true_y = np.array(true_y)[:, 0]
            pred_y = np.array(pred_y)[:, 0]
            comb = itertools.combinations(range(true_y.shape[0]), 2)
            pairs = list()
            # Compute the pairs.
            for _, (i, j) in enumerate(comb):
                if true_y[i] > true_y[j]:
                    pairs.append((i, j))
                elif true_y[i] < true_y[j]:
                    pairs.append((j, i))
            loss = 0.
            pair_num = len(pairs)
            if pair_num == 0:
                return 0.
            for (i, j) in pairs:
                loss += np.log(1 + np.exp(pred_y[j] - pred_y[i]))
            return loss

        # The derivative function.
        def Loss_der(true_y, A, x):
            y_pred = A * np.mat(x).T
            true_y = np.array(true_y)[:, 0]
            pred_y = np.array(y_pred)[:, 0]

            comb = itertools.combinations(range(true_y.shape[0]), 2)
            pairs = list()
            # Compute the pairs.
            for _, (i, j) in enumerate(comb):
                if true_y[i] > true_y[j]:
                    pairs.append((i, j))
                elif true_y[i] < true_y[j]:
                    pairs.append((j, i))
            # Calculate the derivatives.
            grad = np.zeros(A.shape[1])
            pair_num = len(pairs)
            if pair_num == 0:
                return grad
            for (i, j) in pairs:
                e_z = np.exp(pred_y[j] - pred_y[i])
                grad += e_z / (1 + e_z) * (A[j] - A[i]).A1
            return grad

        A, b = pred_y, true_y
        n, m = A.shape
        # Add constraints.
        ineq_cons = {'type': 'ineq',
                     'fun': lambda x: np.array(x),
                     'jac': lambda x: np.eye(len(x))}
        eq_cons = {'type': 'eq',
                   'fun': lambda x: np.array([sum(x) - 1]),
                   'jac': lambda x: np.array([1.] * len(x))}

        x0 = np.array([1. / m] * m)

        def f(x):
            w = np.mat(x).T
            return Loss_func(b, A * w)

        def f_der(x):
            return Loss_der(b, A, x)

        res = minimize(f, x0, method='SLSQP', jac=f_der, constraints=[eq_cons, ineq_cons],
                       options={'ftol': 1e-8, 'disp': False})

        status = False if np.isnan(res.x).any() else True
        if not res.success and status:
            res.x[res.x < 0.] = 0.
            res.x[res.x > 1.] = 1.
            if sum(res.x) > 1.5:
                status = False
        if debug:
            print('the objective', f(res.x))
        return res.x, status

    def get_incumbent(self, num_inc=1):
        assert(len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        return [self.incumbent_configs[i] for i in indices[0:num_inc]], \
               [self.incumbent_obj[i] for i in indices[0: num_inc]]
