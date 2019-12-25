import numpy as np
import time
from mfes.facade.base_facade import BaseFacade
from mfes.config_space import sample_configurations


class RandomSearch(BaseFacade):

    def __init__(self, config_space, objective_func, R, num_iter=50, n_workers=1, random_state=1, method_id="Default"):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers, method_name=method_id)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)
        self.R = R
        self.num_iter = num_iter
        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        self.best_config = None
        self.incumbent_configs = []
        self.incumbent_obj = []

    @BaseFacade.process_manage
    def run(self):
        try:
            for iter in range(self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("Random Search algorithm: %d/%d iteration starts" % (iter, self.num_iter))
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

    def iterate(self):
        configs = sample_configurations(self.config_space, self.num_workers)
        extra_info = None
        ret_val, early_stops = self.run_in_parallel(configs, self.R, extra_info)
        val_losses = [item['loss'] for item in ret_val]

        self.incumbent_configs.extend(configs)
        self.incumbent_obj.extend(val_losses)
        self.add_stage_history(self.stage_id, self.global_incumbent)
        self.stage_id += 1
        self.remove_immediate_model()

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        targets = [self.incumbent_obj[i] for i in indices[0: num_inc]]
        return configs, targets
