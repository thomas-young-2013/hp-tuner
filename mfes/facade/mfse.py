import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from mfes.utils.util_funcs import get_types
from mfes.facade.base_facade import BaseFacade
from mfes.config_space import ConfigurationSpace
from mfes.acquisition_function.acquisition import EI
from mfes.utils.util_funcs import minmax_normalization, std_normalization
from mfes.config_space.util import expand_configurations
from mfes.model.rf_with_instances import RandomForestWithInstances
from mfes.model.weighted_rf_ensemble import WeightedRandomForestCluster
from mfes.config_space import convert_configurations_to_array, sample_configurations

from litebo.utils.history_container import HistoryContainer
from litebo.model.rf_with_instances import RandomForestWithInstances
from litebo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from litebo.optimizer.random_configuration_chooser import ChooserProb


class MFSE(BaseFacade):

    def __init__(self, config_space: ConfigurationSpace, objective_func, R,
                 num_iter=10000, eta=3, n_workers=1, random_state=1,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm', fusion_method='gpoe',
                 power_num=2, method_id='Default'):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers, method_name=method_id)
        self.config_space = config_space
        self.R = R
        self.eta = eta
        self.seed = random_state
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.num_iter = num_iter
        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Parameter for weight method `rank_loss_p_norm`.
        self.power_num = power_num
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.config_space.seed(self.seed)
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        if init_weight is None:
            init_weight = [0.]
            init_weight.extend([1. / self.s_max] * self.s_max)
        assert len(init_weight) == (self.s_max + 1)
        if self.weight_method == 'equal_weight':
            assert self.update_enable is False
        self.logger.info('Weight method & flag: %s-%s' % (self.weight_method, str(self.update_enable)))
        self.logger.info("Initial weight is: %s" % init_weight[:self.s_max + 1])
        types, bounds = get_types(config_space)
        self.num_config = len(bounds)

        self.weighted_surrogate = WeightedRandomForestCluster(
            types, bounds, self.s_max, self.eta, init_weight, self.fusion_method
        )
        self.acquisition_function = EI(model=self.weighted_surrogate)

        self.incumbent_configs = []
        self.incumbent_perfs = []

        self.iterate_id = 0
        self.iterate_r = []
        self.hist_weights = list()

        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = []
            self.target_y[r] = []

        # BO optimizer settings.
        self.configs = list()
        self.history_container = HistoryContainer('mfse-container')
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        rng = np.random.RandomState(seed=random_state)
        self.acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acquisition_function,
            config_space=self.config_space,
            rng=rng,
            max_steps=self.sls_max_steps,
            n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
            n_sls_iterations=self.n_sls_iterations
        )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng=rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.2, rng=rng)

    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):

            if self.update_enable and self.weight_update_id > self.s_max:
                self.update_weight()
            self.weight_update_id += 1

            # Set initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # initial number of iterations per config
            r = int(self.R * self.eta ** (-s))

            # Choose a batch of configurations in different mechanisms.
            start_time = time.time()
            T = self.choose_next_batch(n)
            time_elapsed = time.time() - start_time
            self.logger.info("[%s] Choosing next configurations took %.2f sec." % (self.method_name, time_elapsed))

            extra_info = None
            last_run_num = None

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                n_iter = n_iterations
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iterations

                self.logger.info("MFSE: %d configurations x %d iterations each" %
                                 (int(n_configs), int(n_iterations)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]

                self.target_x[int(n_iterations)].extend(T)
                self.target_y[int(n_iterations)].extend(val_losses)

                if int(n_iterations) == self.R:
                    self.incumbent_configs.extend(T)
                    self.incumbent_perfs.extend(val_losses)
                    # Update history container.
                    for _config, _perf in zip(T, val_losses):
                        self.history_container.add(_config, _perf)

                # Select a number of best configurations for the next loop.
                # Filter out early stops, if any.
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

            for item in self.iterate_r[self.iterate_r.index(r):]:
                # NORMALIZE Objective value: normalization
                normalized_y = std_normalization(self.target_y[item])
                self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                              np.array(normalized_y, dtype=np.float64), r=item)

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("MFSE algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("%d/%d-Iteration took %.2f min." % (iter, self.num_iter, time_elapsed))
                self.iterate_id += 1
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def get_bo_candidates(self, num_configs):
        incumbent = dict()
        incumbent_value = np.min(std_normalization(self.target_y[self.iterate_r[-1]]))
        incumbent['config'] = self.history_container.get_incumbents()[0][1]
        incumbent['obj'] = incumbent_value
        print('Current inc', incumbent)
        # incumbent_value = self.history_container.get_incumbents()[0][1]
        # Update surrogate model in acquisition function.
        self.acquisition_function.update(model=self.weighted_surrogate, eta=incumbent,
                                         num_data=len(self.history_container.data))

        challengers = self.acq_optimizer.maximize(
            runhistory=self.history_container,
            num_points=5000,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return challengers.challengers[:num_configs]

    def choose_next_batch(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            configs = [self.config_space.sample_configuration()]
            configs.extend(sample_configurations(self.config_space, num_config - 1))
            self.configs.extend(configs)
            return configs

        config_candidates = list()
        acq_configs = self.get_bo_candidates(num_configs=2 * num_config)
        acq_idx = 0
        for idx in range(1, 1 + 2 * num_config):
            # Like BOHB, sample a fixed percentage of random configurations.
            if self.random_configuration_chooser.check(idx):
                _config = self.config_space.sample_configuration()
            else:
                _config = acq_configs[acq_idx]
                acq_idx += 1
            if _config not in config_candidates:
                config_candidates.append(_config)
            if len(config_candidates) >= num_config:
                break

        if len(config_candidates) < num_config:
            config_candidates = expand_configurations(config_candidates, self.config_space, num_config)

        _config_candidates = []
        for config in config_candidates:
            if config not in self.configs:  # Check if evaluated
                _config_candidates.append(config)
        self.configs.extend(_config_candidates)
        return _config_candidates

    @staticmethod
    def calculate_ranking_loss(y_pred, y_true):
        length = len(y_pred)
        y_pred = np.reshape(y_pred, -1)
        y_pred1 = np.tile(y_pred, (length, 1))
        y_pred2 = np.transpose(y_pred1)
        diff = y_pred1 - y_pred2
        y_true = np.reshape(y_true, -1)
        y_true1 = np.tile(y_true, (length, 1))
        y_true2 = np.transpose(y_true1)
        y_mask = (y_true1 - y_true2 > 0) + 0
        loss = np.sum(np.log(1 + np.exp(-diff)) * y_mask) / length
        return loss

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def update_weight(self):
        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        r_list = self.weighted_surrogate.surrogate_r
        K = len(r_list)

        if len(test_y) >= 3:
            # Get previous weights
            if self.weight_method in ['rank_loss_softmax', 'rank_loss_single', 'rank_loss_p_norm']:
                preserving_order_p = list()
                preserving_order_nums = list()
                for i, r in enumerate(r_list):
                    fold_num = 5
                    if i != K - 1:
                        mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                        tmp_y = np.reshape(mean, -1)
                        preorder_num, pair_num = MFSE.calculate_preserving_order_num(tmp_y, test_y)
                        preserving_order_p.append(preorder_num / pair_num)
                        preserving_order_nums.append(preorder_num)
                    else:
                        if len(test_y) < 2 * fold_num:
                            preserving_order_p.append(0)
                        else:
                            # 5-fold cross validation.
                            kfold = KFold(n_splits=fold_num)
                            cv_pred = np.array([0] * len(test_y))
                            for train_idx, valid_idx in kfold.split(test_x):
                                train_configs, train_y = test_x[train_idx], test_y[train_idx]
                                valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                                types, bounds = get_types(self.config_space)
                                _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                                _surrogate.train(train_configs, train_y)
                                pred, _ = _surrogate.predict(valid_configs)
                                cv_pred[valid_idx] = pred.reshape(-1)
                            preorder_num, pair_num = MFSE.calculate_preserving_order_num(cv_pred, test_y)
                            preserving_order_p.append(preorder_num / pair_num)
                            preserving_order_nums.append(preorder_num)

                if self.weight_method == 'rank_loss_softmax':
                    order_weight = np.array(np.sqrt(preserving_order_nums))
                    trans_order_weight = order_weight - np.max(order_weight)
                    # Softmax mapping.
                    new_weights = np.exp(trans_order_weight) / sum(np.exp(trans_order_weight))
                elif self.weight_method == 'rank_loss_p_norm':
                    trans_order_weight = np.array(preserving_order_p)
                    power_sum = np.sum(np.power(trans_order_weight, self.power_num))
                    new_weights = np.power(trans_order_weight, self.power_num) / power_sum
                else:
                    _idx = np.argmax(np.array(preserving_order_nums))
                    new_weights = [0.] * K
                    new_weights[_idx] = 1.
            elif self.weight_method == 'rank_loss_prob':
                # For basic surrogate i=1:K-1.
                mean_list, var_list = list(), list()
                for i, r in enumerate(r_list[:-1]):
                    mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                    mean_list.append(np.reshape(mean, -1))
                    var_list.append(np.reshape(var, -1))
                sample_num = 100
                min_probability_array = [0] * K
                for _ in range(sample_num):
                    order_preseving_nums = list()

                    # For basic surrogate i=1:K-1.
                    for idx in range(K - 1):
                        sampled_y = np.random.normal(mean_list[idx], var_list[idx])
                        _num, _ = MFSE.calculate_preserving_order_num(sampled_y, test_y)
                        order_preseving_nums.append(_num)

                    fold_num = 5
                    # For basic surrogate i=K. cv
                    if len(test_y) < 2 * fold_num:
                        order_preseving_nums.append(0)
                    else:
                        # 5-fold cross validation.
                        kfold = KFold(n_splits=fold_num)
                        cv_pred = np.array([0] * len(test_y))
                        for train_idx, valid_idx in kfold.split(test_x):
                            train_configs, train_y = test_x[train_idx], test_y[train_idx]
                            valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                            types, bounds = get_types(self.config_space)
                            _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                            _surrogate.train(train_configs, train_y)
                            _pred, _var = _surrogate.predict(valid_configs)
                            sampled_pred = np.random.normal(_pred.reshape(-1), _var.reshape(-1))
                            cv_pred[valid_idx] = sampled_pred
                        _num, _ = MFSE.calculate_preserving_order_num(cv_pred, test_y)
                        order_preseving_nums.append(_num)
                    max_id = np.argmax(order_preseving_nums)
                    min_probability_array[max_id] += 1
                new_weights = np.array(min_probability_array) / sample_num

            elif self.weight_method == 'opt_based':
                mean_list, var_list = list(), list()
                for i, r in enumerate(r_list):
                    if i != K - 1:
                        mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                        tmp_y = np.reshape(mean, -1)
                        tmp_var = np.reshape(var, -1)
                        mean_list.append(tmp_y)
                        var_list.append(tmp_var)
                    else:
                        if len(test_y) < 8:
                            mean_list.append(np.array([0] * len(test_y)))
                            var_list.append(np.array([0] * len(test_y)))
                        else:
                            # 5-fold cross validation.
                            kfold = KFold(n_splits=5)
                            cv_pred = np.array([0] * len(test_y))
                            cv_var = np.array([0] * len(test_y))
                            for train_idx, valid_idx in kfold.split(test_x):
                                train_configs, train_y = test_x[train_idx], test_y[train_idx]
                                valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                                types, bounds = get_types(self.config_space)
                                _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                                _surrogate.train(train_configs, train_y)
                                pred, var = _surrogate.predict(valid_configs)
                                cv_pred[valid_idx] = pred.reshape(-1)
                                cv_var[valid_idx] = var.reshape(-1)
                            mean_list.append(cv_pred)
                            var_list.append(cv_var)
                means = np.array(mean_list)
                vars = np.array(var_list) + 1e-8

                def min_func(x):
                    x = np.reshape(np.array(x), (1, len(x)))
                    ensemble_vars = 1 / (x @ (1 / vars))
                    ensemble_means = x @ (means / vars) * ensemble_vars
                    ensemble_means = np.reshape(ensemble_means, -1)
                    self.logger.info("Loss:" + str(x))
                    return MFSE.calculate_ranking_loss(ensemble_means, test_y)

                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                               {'type': 'ineq', 'fun': lambda x: x - 0},
                               {'type': 'ineq', 'fun': lambda x: 1 - x}]
                res = minimize(min_func, np.array([1e-8] * K), constraints=constraints)
                new_weights = res.x
            else:
                raise ValueError('Invalid weight method: %s!' % self.weight_method)
        else:
            old_weights = list()
            for i, r in enumerate(r_list):
                _weight = self.weighted_surrogate.surrogate_weight[r]
                old_weights.append(_weight)
            new_weights = old_weights.copy()

        self.logger.info('[%s] %d-th Updating weights: %s' % (
            self.weight_method, self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        for i, r in enumerate(r_list):
            self.weighted_surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        self.hist_weights.append(new_weights)
        np.save('data/%s_weights_%s.npy' % (self.method_name, self.method_name), np.asarray(self.hist_weights))

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        return [self.incumbent_configs[i] for i in indices[0:num_inc]], \
               [self.incumbent_perfs[i] for i in indices[0: num_inc]]

    def get_weights(self):
        return self.hist_weights
