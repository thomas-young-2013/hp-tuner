import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from mfes.utils.util_funcs import get_types
from mfes.facade.base_facade import BaseFacade
from mfes.config_space import ConfigurationSpace
from mfes.acquisition_function.acquisition import EI
from mfes.utils.util_funcs import minmax_normalization
from mfes.config_space.util import expand_configurations
from mfes.optimizer.random_sampling import RandomSampling
from mfes.model.rf_with_instances import RandomForestWithInstances
from mfes.model.weighted_rf_ensemble import WeightedRandomForestCluster
from mfes.config_space import convert_configurations_to_array, sample_configurations


class MFSE(BaseFacade):

    def __init__(self, config_space: ConfigurationSpace, objective_func, R,
                 num_iter=10000, eta=3, p=0.5, n_workers=1, random_state=1,
                 init_weight=None, update_enable=True, weight_method='rank_loss_softmax',
                 multi_surrogate=True, fusion_method='gpoe'):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.config_space = config_space
        self.p = p
        self.R = R
        self.eta = eta
        self.seed = random_state
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.num_iter = num_iter
        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.config_space.seed(self.seed)
        self.weight_update_id = 0
        self.multi_surrogate = multi_surrogate

        if init_weight is None:
            init_weight = [1. / (self.s_max + 1)] * (self.s_max + 1)
        self.logger.info("Initial weight is: %s" % init_weight[:self.s_max + 1])
        types, bounds = get_types(config_space)
        self.num_config = len(bounds)

        self.weighted_surrogate_update = WeightedRandomForestCluster(
            types, bounds, self.s_max, self.eta, init_weight, self.fusion_method
        )
        self.weighted_surrogate = WeightedRandomForestCluster(
            types, bounds, self.s_max, self.eta, init_weight, self.fusion_method
        )
        self.weighted_acquisition_func = EI(model=self.weighted_surrogate)
        self.weighted_acq_optimizer = RandomSampling(self.weighted_acquisition_func,
                                                     config_space, n_samples=max(500, 50 * self.num_config))

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
            T = self.choose_next_weighted(n)
            time_elapsed = time.time() - start_time
            self.logger.info("Choosing next configurations took %.2f sec." % time_elapsed)

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
                normalized_y = minmax_normalization(self.target_y[item])
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
                self.logger.info("Iteration took %.2f min." % time_elapsed)
                self.iterate_id += 1
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def choose_next_weighted(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            return sample_configurations(self.config_space, num_config)

        config_cnt = 0
        config_candidates = list()
        total_sample_cnt = 0

        while config_cnt < num_config and total_sample_cnt < 3 * num_config:
            incumbent = dict()
            max_r = self.iterate_r[-1]
            best_index = np.argmin(self.target_y[max_r])
            incumbent['config'] = self.target_x[max_r][best_index]
            approximate_obj = self.weighted_surrogate.predict(convert_configurations_to_array([incumbent['config']]))[0]
            incumbent['obj'] = approximate_obj

            self.weighted_acquisition_func.update(model=self.weighted_surrogate, eta=incumbent)
            _config = self.weighted_acq_optimizer.maximize(batch_size=1)[0]

            if _config not in config_candidates:
                config_candidates.append(_config)
                config_cnt += 1
            total_sample_cnt += 1

        if config_cnt < num_config:
            config_candidates = expand_configurations(config_candidates, self.config_space, num_config)
        return config_candidates

    @staticmethod
    def calculate_ranking_loss(self, y_pred, y_true):
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
    def calculate_preserving_order_num(self, y_pred, y_true, preserve_order=True):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx+1, array_size):
                if y_true[idx] > y_true[inner_idx] and y_pred[idx] > y_pred[inner_idx]:
                    order_preserving_num += 1
                total_pair_num += 1
        if preserve_order:
            return order_preserving_num
        else:
            return total_pair_num - order_preserving_num

    def update_weight(self):
        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        # Get previous weights
        r_list = self.weighted_surrogate.surrogate_r
        K = len(r_list)

        if self.weight_method == 'rank_loss_softmax':
            # Get means and vars
            mean_list = list()
            preserving_order_nums = list()
            for i, r in enumerate(r_list):
                mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                tmp_y = np.reshape(mean, -1)
                mean_list.append(tmp_y)
                preserving_order_nums.append(self.calculate_preserving_order_num(tmp_y, test_y))
            order_weight = np.array(np.sqrt(preserving_order_nums))
            trans_order_weight = order_weight - np.max(order_weight)
            # Softmax mapping.
            new_weights = np.exp(trans_order_weight) / sum(np.exp(trans_order_weight))
        # TODO: Add more weight learning mehtods here.
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
                    _num = self.calculate_preserving_order_num(sampled_y, test_y)
                    order_preseving_nums.append(_num)
                # For basic surrogate i=K. cv
                if len(test_y) < 10:
                    order_preseving_nums.append(0)
                else:
                    # 5-fold cross validation.
                    kfold = KFold(n_splits=5)
                    cv_pred = np.array([0]*len(test_y))
                    for train_idx, valid_idx in kfold.split(test_x):
                        train_configs, train_y = test_x[train_idx], test_y[train_idx]
                        valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                        types, bounds = get_types(self.config_space)
                        _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                        _surrogate.train(train_configs, train_y)
                        pred = _surrogate.predict(valid_configs)
                        cv_pred[valid_idx] = pred.reshape(-1)
                    _num = self.calculate_preserving_order_num(cv_pred, test_y)
                    order_preseving_nums.append(_num)
                max_id = np.argmax(order_preseving_nums)
                min_probability_array[max_id] += 1
            new_weights = np.array(min_probability_array) / sample_num
        else:
            raise ValueError('Invalid weight method: %s!' % self.weight_method)

        self.logger.info('Updating weights: %s' % str(new_weights))

        # Assign the weight to each basic surrogate.
        for i, r in enumerate(r_list):
            self.weighted_surrogate.surrogate_weight[r] = new_weights[i]

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
