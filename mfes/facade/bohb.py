import time
import random
import numpy as np
from math import log, ceil

from mfes.config_space import ConfigurationSpace
from mfes.model.rf_with_instances import RandomForestWithInstances
from mfes.utils.util_funcs import get_types
from mfes.acquisition_function.acquisition import EI
from mfes.optimizer.random_sampling import RandomSampling
from mfes.config_space import convert_configurations_to_array, sample_configurations
from mfes.config_space.util import expand_configurations
from mfes.facade.base_facade import BaseFacade


class BOHB(BaseFacade):
    """ The implementation of BOHB.
        The paper can be found in https://arxiv.org/abs/1807.01774 .
    """

    def __init__(self, config_space: ConfigurationSpace, objective_func, R,
                 num_iter=10, eta=3, p=0.3, n_workers=1, random_state=1):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.config_space = config_space
        self.seed = random_state
        self.config_space.seed(self.seed)
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

    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):
            # Set initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # Set initial number of iterations per config
            r = self.R * self.eta ** (-s)
            
            # Sample n configurations according to BOHB strategy.
            T = self.choose_next(n)
            extra_info = None
            last_run_num = None
            for i in range((s + 1) - int(skip_last)): # changed from s + 1
                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations.

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                n_iter = n_iterations
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iterations

                self.logger.info("BOHB: %d configurations x %d iterations each" % (int(n_configs), int(n_iterations)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]

                if int(n_iterations) == self.R:
                    self.incumbent_configs.extend(T)
                    self.incumbent_obj.extend(val_losses)
                
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

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(self.num_iter):
                self.logger.info('-'*50)
                self.logger.info("BOHB algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time)/60
                self.logger.info("Iteration took %.2f min." % time_elapsed)
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # Clean the immediate result.
            self.remove_immediate_model()

    def choose_next(self, num_config):
        if len(self.incumbent_obj) < 2 * self.num_config:
            return sample_configurations(self.config_space, num_config)
        
        self.logger.info('Train feature is: %s' % str(self.incumbent_configs[:5]))
        self.logger.info('Train target is: %s' % str(self.incumbent_obj))
        self.surrogate.train(convert_configurations_to_array(self.incumbent_configs),
                             np.array(self.incumbent_obj, dtype=np.float64))

        config_cnt = 0
        total_sample_cnt = 0
        config_candidates = []
        while config_cnt < num_config and total_sample_cnt < 3 * num_config:
            if random.random() < self.p:
                rand_config = self.config_space.sample_configuration(1)
            else:
                # print('use surrogate to produce candidate.')
                incumbent = dict()
                best_index = np.argmin(self.incumbent_obj)
                incumbent['obj'] = self.incumbent_obj[best_index]
                incumbent['config'] = self.incumbent_configs[best_index]

                self.acquisition_func.update(model=self.surrogate, eta=incumbent)
                rand_config = self.acq_optimizer.maximize(batch_size=1)[0]
            if rand_config not in config_candidates:
                config_candidates.append(rand_config)
                config_cnt += 1
            total_sample_cnt += 1
        if config_cnt < num_config:
            config_candidates = expand_configurations(config_candidates, self.config_space, num_config)
        return config_candidates

    def get_incumbent(self, num_inc=1):
        assert(len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        targets = [self.incumbent_obj[i] for i in indices[0: num_inc]]
        return configs, targets
