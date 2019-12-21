import numpy as np
import time
import hashlib
from scipy.stats import norm
from math import log, ceil
from mfes.model.rf_with_instances import RandomForestWithInstances
from mfes.utils.util_funcs import get_types
from mfes.acquisition_function.acquisition import EI
from mfes.optimizer.random_sampling import RandomSampling
from mfes.config_space import convert_configurations_to_array, sample_configurations
from mfes.config_space.util import expand_configurations
from mfes.facade.base_facade import BaseFacade
from mfes.config_space.util import get_configuration_id
from mfes.model.lcnet import LC_ES


class SMAC_ES(BaseFacade):

    def __init__(self, config_space, objective_func, R,
                 num_iter=10, n_workers=1, eta=3, es_gap=9, rho=0.7):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers, need_lc=True)
        self.config_space = config_space
        self.R = R
        self.num_iter = num_iter
        self.eta = eta
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(R))
        self.inner_iteration_n = (self.s_max + 1) * (self.s_max + 1)

        types, bounds = get_types(config_space)
        self.num_config = len(bounds)
        self.surrogate = RandomForestWithInstances(types=types, bounds=bounds)
        self.acquisition_func = EI(model=self.surrogate)
        # TODO: add SMAC's optimization algorithm.
        self.acq_optimizer = RandomSampling(self.acquisition_func, config_space, n_samples=max(500, 50*self.num_config))

        self.incumbent_configs = []
        self.incumbent_obj = []

        self.lcnet_model = LC_ES()
        self.early_stop_gap = es_gap
        self.es_rho = rho
        self.lc_training_x = None
        self.lc_training_y = None

    def iterate(self):
        for _ in range(self.inner_iteration_n):
            T = self.choose_next(self.num_workers)

            extra_info = None
            lc_info = dict()
            lc_conf_mapping = dict()

            # assume no same configuration in the same batch: T
            for item in T:
                conf_id = get_configuration_id(item.get_dictionary())
                sha = hashlib.sha1(conf_id.encode('utf8'))
                conf_id = sha.hexdigest()
                lc_conf_mapping[conf_id] = item

            total_iter_num = self.R // self.early_stop_gap
            for iter_num in range(1, 1+total_iter_num):
                self.logger.info('start iteration gap %d' % iter_num)
                ret_val, early_stops = self.run_in_parallel(T, self.early_stop_gap, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]
                for item in ret_val:
                    conf_id = item['ref_id']
                    if not self.restart_needed:
                        if conf_id not in lc_info:
                            lc_info[conf_id] = []
                        lc_info[conf_id].extend(item['lc_info'])
                    else:
                        lc_info[conf_id] = item['lc_info']

                if iter_num == total_iter_num:
                    self.incumbent_configs.extend(T)
                    self.incumbent_obj.extend(val_losses)

                T = [config for i, config in enumerate(T) if not early_stops[i]]
                extra_info = [ref for i, ref in enumerate(ref_list) if not early_stops[i]]
                if len(T) == 0:
                    break

                if len(self.incumbent_obj) >= 2*self.num_config and iter_num != total_iter_num:
                    # learning curve based early stop strategy.
                    ref_list = extra_info
                    early_stops = self.stop_early(T)
                    T = [config for i, config in enumerate(T) if not early_stops[i]]
                    extra_info = [ref for i, ref in enumerate(ref_list) if not early_stops[i]]
                if len(T) == 0:
                    break

            # keep learning curve data
            for item, config in lc_conf_mapping.items():
                lc_data = lc_info[item]
                if len(lc_data) > 0:
                    n_epochs = len(lc_data)
                    # self.logger.info('insert one learning curve data into dataset.')
                    t_idx = np.arange(1, n_epochs + 1) / n_epochs
                    conf_data = convert_configurations_to_array([config])
                    x = np.repeat(conf_data, t_idx.shape[0], axis=0)
                    x = np.concatenate((x, t_idx[:, None]), axis=1)
                    y = np.array(lc_data)
                    if self.lc_training_x is None:
                        self.lc_training_x, self.lc_training_y = x, y
                    else:
                        self.lc_training_x = np.concatenate((self.lc_training_x, x), 0)
                        self.lc_training_y = np.concatenate((self.lc_training_y, y), 0)
            self.logger.info('training data shape: %s' % str(self.lc_training_x.shape))
            if len(self.incumbent_obj) >= 2*self.num_config and len(self.incumbent_obj) % self.num_config == 0:
                self.lcnet_model.train(self.lc_training_x, self.lc_training_y)

            self.add_stage_history(self.stage_id, self.global_incumbent)
            self.stage_id += 1
            self.remove_immediate_model()

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(self.num_iter):
                self.logger.info('-'*50)
                self.logger.info("SMAC with ES algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time)/60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def stop_early(self, T):
        configs_data = convert_configurations_to_array(T)
        x_test = None
        for i in range(configs_data.shape[0]):
            x = np.concatenate((configs_data[i, None, :], np.array([[1.0]])), axis=1)
            if x_test is None:
                x_test = x
            else:
                x_test = np.concatenate((x_test, x), 0)

        m, v = self.lcnet_model.predict(x_test)
        best_accuracy = 1 - self.global_incumbent
        s = np.sqrt(v)
        less_p = norm.cdf((best_accuracy-m)/s)
        self.logger.info('early stop prob: %s' % str(less_p))
        early_stop_flag = (less_p >= self.es_rho)

        self.logger.info('early stop vector: %s' % str(early_stop_flag))

        for i, flag in enumerate(early_stop_flag):
            if flag:
                self.incumbent_configs.append(T[i])
                self.incumbent_obj.append(m[i])
        return early_stop_flag

    def choose_next(self, num_config):
        if len(self.incumbent_obj) < 2*self.num_config:
            return sample_configurations(self.config_space, num_config)

        # print('choose next starts!')
        self.logger.info('train feature is: %s' % str(self.incumbent_configs[-5:]))
        self.logger.info('train target is: %s' % str(self.incumbent_obj))

        self.surrogate.train(convert_configurations_to_array(self.incumbent_configs),
                             np.array(self.incumbent_obj, dtype=np.float64))

        conf_cnt = 0
        total_cnt = 0
        next_configs = []
        while conf_cnt < num_config and total_cnt < 5*num_config:
            incumbent = dict()
            best_index = np.argmin(self.incumbent_obj)
            incumbent['obj'] = self.incumbent_obj[best_index]
            incumbent['config'] = self.incumbent_configs[best_index]

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
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        targets = [self.incumbent_obj[i] for i in indices[0: num_inc]]
        return configs, targets
