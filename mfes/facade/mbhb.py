import numpy as np
import time
import hashlib
from mfes.utils.util_funcs import get_types
from mfes.config_space import convert_configurations_to_array, sample_configurations
from mfes.config_space.util import expand_configurations
from mfes.facade.base_facade import BaseFacade
from math import log, ceil
from mfes.config_space.util import get_configuration_id
from mfes.model.lcnet import LC_ES


class MBHB(BaseFacade):

    def __init__(self, config_space, objective_func, R,
                 num_iter=10, eta=3, p=0.5, n_workers=1):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers, need_lc=True)
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

        self.incumbent_configs = []
        self.incumbent_obj = []

        self.lcnet_model = LC_ES()
        self.lc_training_x = None
        self.lc_training_y = None

    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.R * self.eta ** (-s)

            # n random configurations
            T = self.choose_next(n)
            extra_info = None
            last_run_num = None

            lc_info = dict()
            lc_conf_mapping = dict()
            # assume no same configuration in the same batch: T
            for item in T:
                conf_id = get_configuration_id(item.get_dictionary())
                sha = hashlib.sha1(conf_id.encode('utf8'))
                conf_id = sha.hexdigest()
                lc_conf_mapping[conf_id] = item

            for i in range((s + 1) - int(skip_last)):

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                n_iter = n_iterations
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iterations

                self.logger.info("MBHB: %d configurations x %d iterations each" % (int(n_configs), int(n_iterations)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)

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

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("MBHB algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def choose_next(self, num_config):
        self.logger.info('LCNet: model-based choosing.')
        if len(self.incumbent_obj) <= 0:
            return sample_configurations(self.config_space, num_config)
        self.logger.info('start to training LCNet, training data shape: %s' % str(self.lc_training_x.shape))
        self.lcnet_model.train(self.lc_training_x, self.lc_training_y)

        next_configs = []
        random_configs = sample_configurations(self.config_space, 50 * self.num_config)
        random_configs_data = convert_configurations_to_array(random_configs)
        x_test = None
        for i in range(random_configs_data.shape[0]):
            x = np.concatenate((random_configs_data[i, None, :], np.array([[1.0]])), axis=1)
            if x_test is None:
                x_test = x
            else:
                x_test = np.concatenate((x_test, x), 0)
        m, v = self.lcnet_model.predict(x_test)
        sorted_configs = [random_configs[i] for i in np.argsort(-m)]
        print(sorted_configs[:5])
        number_flag = False
        for config in sorted_configs:
            if config not in next_configs:
                next_configs.append(config)
            if len(next_configs) == num_config:
                number_flag = True
                break
        if not number_flag:
            next_configs = expand_configurations(next_configs, self.config_space, num_config)
            self.logger.warning('MBHB: add random configuration here.' + '=' * 50)
        return next_configs

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        targets = [self.incumbent_obj[i] for i in indices[0: num_inc]]
        return configs, targets
