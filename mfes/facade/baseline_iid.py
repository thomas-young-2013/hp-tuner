import numpy as np
import random
import time
from mfes.model.rf_with_instances import RandomForestWithInstances
from mfes.utils.util_funcs import get_types
from mfes.acquisition_function.acquisition import EI
from mfes.optimizer.random_sampling import RandomSampling
from mfes.config_space import convert_configurations_to_array, sample_configurations
from mfes.facade.base_facade import BaseFacade
from mfes.config_space.util import expand_configurations
from mfes.utils.util_funcs import minmax_normalization
from math import log, ceil


class BaseIID(BaseFacade):

    def __init__(self, config_space, objective_func, R,
                 num_iter=10, eta=3, p=0.5, n_workers=1):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.config_space = config_space
        self.p = p
        self.R = R
        self.eta = eta
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.num_iter = num_iter

        types, bounds = get_types(config_space)
        self.num_config = len(bounds)
        self.surrogate = RandomForestWithInstances(types=types, bounds=bounds)
        self.acquisition_func = EI(model=self.surrogate)
        self.acq_optimizer = RandomSampling(self.acquisition_func, config_space, n_samples=max(500, 50*self.num_config))

        self.incumbent_configs = []
        self.incumbent_obj = []
        self.iterate_id = 0
        self.iterate_r = []

        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max+1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = []
            self.target_y[r] = []

    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # initial number of iterations per config
            r = int(self.R * self.eta ** (-s))

            # choose a batch of configurations in different mechanisms.
            start_time = time.time()
            T = self.choose_next(n)
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

                self.logger.info("Baseline: %d configurations x %d iterations each" %
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

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-'*50)
                self.logger.info("baseline-iid algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time)/60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.iterate_id += 1
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def choose_next(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            return sample_configurations(self.config_space, num_config)

        train_x = []
        train_y = []
        for item in self.iterate_r:
            # objective value normalization: min-max linear normalization
            normalized_y = minmax_normalization(self.target_y[item])
            train_y.extend(normalized_y)
            train_x.extend(self.target_x[item])

        self.surrogate.train(convert_configurations_to_array(train_x),
                                      np.array(train_y, dtype=np.float64))

        self.logger.info('train feature is: %s' % str(train_x[:10]))
        self.logger.info('train data size is: %d' % len(train_y))

        conf_cnt = 0
        next_configs = []
        total_cnt = 0

        while conf_cnt < num_config and total_cnt < 2 * num_config:
            incumbent = dict()
            max_r = self.iterate_r[-1]
            best_index = np.argmin(self.target_y[max_r])
            incumbent['config'] = self.target_x[max_r][best_index]
            approximate_obj = self.surrogate.predict(convert_configurations_to_array([incumbent['config']]))[0]
            incumbent['obj'] = approximate_obj

            self.acquisition_func.update(model=self.surrogate, eta=incumbent)
            rand_config = self.acq_optimizer.maximize(batch_size=1)[0]

            if rand_config not in next_configs:
                next_configs.append(rand_config)
                conf_cnt += 1
            total_cnt += 1

        if conf_cnt < num_config:
            next_configs = expand_configurations(next_configs, self.config_space, num_config)
        return next_configs

    def get_incumbent(self, num_inc=1):
        assert(len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        return [self.incumbent_configs[i] for i in indices[0:num_inc]], \
               [self.incumbent_obj[i] for i in indices[0: num_inc]]
