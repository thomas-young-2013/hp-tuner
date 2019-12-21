import numpy as np
import time
from mfes.facade.base_facade import BaseFacade
from mfes.config_space import sample_configurations
from math import log, ceil


class Hyperband(BaseFacade):

    def __init__(self, config_space, objective_func, R, num_iter=10, n_workers=1):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.configuration_space = config_space

        self.num_iter = num_iter
        self.max_iter = R  	    # maximum iterations per configuration
        self.eta = 3			# defines configuration downsampling rate (default = 3)
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.incumbent_configs = []
        self.incumbent_obj = []

    # can be called multiple times
    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            # T = self.configuration_space.sample_configuration(n)
            T = sample_configurations(self.configuration_space, n)
            incumbent_loss = np.inf
            extra_info = None
            last_run_num = None
            for i in range((s + 1) - int(skip_last)): # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)
                n_iter = n_iterations
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iterations

                self.logger.info("HB: %d configurations x %d iterations each" % (int(n_configs), int(n_iterations)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]

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
            if not np.isnan(incumbent_loss):
                self.incumbent_configs.append(T[0])
                self.incumbent_obj.append(incumbent_loss)
            self.remove_immediate_model()

    @BaseFacade.process_manage
    def run(self, skip_last=0):
        try:
            for iter in range(self.num_iter):
                self.logger.info('-'*50)
                self.logger.info("HB algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate(skip_last=skip_last)
                time_elapsed = (time.time() - start_time)/60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.save_intemediate_statistics()
            for i, obj in enumerate(self.incumbent_obj):
                self.logger.info('%dth config: %s, obj: %f' % (i+1, str(self.incumbent_configs[i]), self.incumbent_obj[i]))
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

    def get_incumbent(self, num_inc=1):
        assert(len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        return [self.incumbent_configs[i] for i in indices[0:num_inc]], \
               [self.incumbent_obj[i] for i in indices[0: num_inc]]
