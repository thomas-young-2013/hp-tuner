import numpy as np
import random
import time
from scipy.optimize import minimize
from hoist.model.rf_with_instances import RandomForestWithInstances
from hoist.model.weighted_rf_ensemble import WeightedRandomForestCluster
from hoist.utils.util_funcs import get_types
from hoist.acquisition_function.acquisition import EI
from hoist.optimizer.random_sampling import RandomSampling
from hoist.config_space import convert_configurations_to_array, sample_configurations
from hoist.facade.base_facade import BaseFacade
from hoist.config_space.util import expand_configurations
from hoist.utils.util_funcs import minmax_normalization
from math import log, ceil


# TODO: mechanism of random forest optimizer.
# TODO: the hyperparameter of random forest.
# TODO: weight decay.
class MFSE(BaseFacade):

    def __init__(self, config_space, objective_func, R,
                 num_iter=10, eta=3, p=0.5, n_workers=1, init_weight=None,
                 update_enable=False, random_mode=True, multi_surrogate=True):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.config_space = config_space
        self.p = p
        self.R = R
        self.eta = eta
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.num_iter = num_iter
        self.update_enable = update_enable
        self.random_mode = random_mode

        self.weight_update_id = 0
        self.multi_surrogate = multi_surrogate

        if init_weight is None:
            init_weight = [1. / (self.s_max + 1)] * (self.s_max + 1)
        self.logger.info("initial confidence weight %s" % init_weight[:self.s_max + 1])
        types, bounds = get_types(config_space)
        self.num_config = len(bounds)
        self.surrogate = RandomForestWithInstances(types=types, bounds=bounds)
        self.acquisition_func = EI(model=self.surrogate)
        self.acq_optimizer = RandomSampling(self.acquisition_func, config_space,
                                            n_samples=max(500, 50 * self.num_config))

        self.weighted_surrogate = WeightedRandomForestCluster(types, bounds, self.s_max, self.eta, init_weight,
                                                              'lc')
        self.weighted_acquisition_func = EI(model=self.weighted_surrogate)
        self.weighted_acq_optimizer = RandomSampling(self.weighted_acquisition_func,
                                                     config_space, n_samples=max(500, 50 * self.num_config))

        self.incumbent_configs = []
        self.incumbent_obj = []
        self.init_tradeoff = 0.5
        self.tradeoff_dec_rate = 0.8
        self.iterate_id = 0
        self.iterate_r = []
        self.hist_weights = list()

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
                self.update_weight_vector()
            self.weight_update_id += 1

            # initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # initial number of iterations per config
            r = int(self.R * self.eta ** (-s))

            # choose a batch of configurations in different mechanisms.
            start_time = time.time()
            T = self.choose_next_weighted(n)
            time_elapsed = time.time() - start_time
            self.logger.info("choosing next configurations took %.2f sec." % time_elapsed)

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
                    self.incumbent_obj.extend(val_losses)
                # select a number of best configurations for the next loop
                # filter out early stops, if any
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
                # objective value normalization: min-max linear normalization
                normalized_y = minmax_normalization(self.target_y[item])
                self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                              np.array(normalized_y, dtype=np.float64), r=item)
        # TODO: trade off value: decay (bayesian optimization did? do we need trade off e&e again?)
        self.init_tradeoff *= self.tradeoff_dec_rate

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("MFSE algorithm: %d/%d iteration starts" % (iter, self.num_iter))
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

    def choose_next_weighted(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            return sample_configurations(self.config_space, num_config)

        conf_cnt = 0
        next_configs = []
        total_cnt = 0
        while conf_cnt < num_config and total_cnt < 2 * num_config:
            # in Bayesian optimization, eliminate epsilon sampling.
            incumbent = dict()
            # TODO: problem-->use the best in maximal resource.
            # TODO: smac's optmization algorithm.
            max_r = self.iterate_r[-1]
            best_index = np.argmin(self.target_y[max_r])
            incumbent['config'] = self.target_x[max_r][best_index]
            approximate_obj = self.weighted_surrogate.predict(convert_configurations_to_array([incumbent['config']]))[0]
            incumbent['obj'] = approximate_obj

            self.weighted_acquisition_func.update(model=self.weighted_surrogate, eta=incumbent)
            rand_config = self.weighted_acq_optimizer.maximize(batch_size=1)[0]

            if rand_config not in next_configs:
                next_configs.append(rand_config)
                conf_cnt += 1
            total_cnt += 1

        if conf_cnt < num_config:
            next_configs = expand_configurations(next_configs, self.config_space, num_config)
        return next_configs

    # mean ranking loss
    # def _calculate_loss(self, y_pred, y_true):
    #     length = len(y_pred)
    #     y_pred = np.reshape(y_pred, -1)
    #     y_pred1 = np.tile(y_pred, (length, 1))
    #     y_pred2 = np.transpose(y_pred1)
    #     diff = y_pred1 - y_pred2
    #     y_true = np.reshape(y_true, -1)
    #     y_true1 = np.tile(y_true, (length, 1))
    #     y_true2 = np.transpose(y_true1)
    #     y_mask = (y_true1 - y_true2 > 0) + 0
    #     loss = np.sum(np.log(1 + np.exp(-diff)) * y_mask) / length
    #     return loss

    # Ordered pair (divide-and-conquer)
    def _ordered_pair(self, y_pred, y_true):
        length = len(y_pred)
        sorted_idx = np.argsort(y_true)
        sorted_pred = [y_pred[i] for i in sorted_idx]

        def inverted_pair(lst):
            if len(lst) == 1:
                return lst, 0
            else:
                n = len(lst) // 2
                lst1, count1 = inverted_pair(lst[0:n])
                lst2, count2 = inverted_pair(lst[n:len(lst)])
                i = j = cnt = 0
                res = []
                while i < len(lst1) and j < len(lst2):
                    if lst1[i] <= lst2[j]:
                        res.append(lst1[i])
                        i += 1
                    else:
                        res.append(lst2[j])
                        cnt += len(lst1) - i
                        j += 1
                res += lst1[i:]
                res += lst2[j:]
                return lst, count1 + count2 + cnt

        return int(length * (length - 1) / 2 - inverted_pair(sorted_pred)[1])

    def update_weight_vector(self):
        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = minmax_normalization(self.target_y[max_r])
        test_y = np.array(test_y)

        # Get previous weights
        r_list = self.weighted_surrogate.surrogate_r
        cur_confidence = self.weighted_surrogate.surrogate_weight

        # Get means and vars
        mean_list = []
        # var_list = []
        # loss_list = []
        order_weight = []
        for i, r in enumerate(r_list):
            mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
            tmp_y = np.reshape(mean, -1)
            # tmp_var = np.reshape(var, -1)
            mean_list.append(tmp_y)
            # var_list.append(tmp_var)
            # loss_list.append(self._calculate_loss(tmp_y, test_y))
            order_weight.append(self._ordered_pair(tmp_y, test_y))

        order_weight = np.array(order_weight)
        order_weight = np.exp(order_weight) / sum(np.exp(order_weight))  # Softmax
        self.logger.info(order_weight)
        # means = np.array(mean_list)
        # vars = np.array(var_list) + 1e-8

        # if self.multi_surrogate:
        #     def min_func(x):
        #         x = np.reshape(np.array(x), (1, len(x)))
        #         ensemble_vars = 1 / (x @ (1 / vars))
        #         ensemble_means = x @ (means / vars) * ensemble_vars
        #         ensemble_means = np.reshape(ensemble_means, -1)
        #         return self._calculate_loss(ensemble_means, test_y)
        #
        #     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        #                    {'type': 'ineq', 'fun': lambda x: x - 0},
        #                    {'type': 'ineq', 'fun': lambda x: 1 - x}]
        #     res = minimize(min_func, curr_list, constraints=constraints)

        updated_weights = list()
        max_surrogate_id = np.argmax(order_weight)
        for i, r in enumerate(r_list):
            if not self.multi_surrogate:
                self.weighted_surrogate.surrogate_weight[r] = 1.0 if i == max_surrogate_id else 0
            else:
                self.weighted_surrogate.surrogate_weight[r] = order_weight[i]
            updated_weights.append(self.weighted_surrogate.surrogate_weight[r])
            self.logger.info('update surrogate weight:%d-%.4f' % (r, self.weighted_surrogate.surrogate_weight[r]))
        self.hist_weights.append(updated_weights)
        np.save('data/tmp_weights_%s.npy' % self.method_name, np.asarray(self.hist_weights))

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        return [self.incumbent_configs[i] for i in indices[0:num_inc]], \
               [self.incumbent_obj[i] for i in indices[0: num_inc]]

    def get_weights(self):
        return self.hist_weights
