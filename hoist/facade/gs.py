import numpy as np
import itertools
import time

from hoist.facade.base_facade import BaseFacade


class GridSearch(BaseFacade):

    def __init__(self, objective_func, R, n_workers=1, eta=3, case=1):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.R = R
        self.eta = eta
        self.case = case

    def iterate(self):
        case = self.case
        if case == 1:
            n1 = np.linspace(0.1, 1.0, num=30)
            n2 = pow(10, np.linspace(-7, -2, num=30))
            hp1, hp2 = 'keep_prob', 'lr'
        elif case == 2:
            n1 = np.linspace(64, 768, num=30)
            n2 = np.linspace(0.1, 1.0, num=30)
            hp1, hp2 = 'fc_unit', 'keep_prob'
        elif case == 3:
            n1 = pow(10, np.linspace(-7, -2, num=30))
            n2 = np.linspace(16, 256, num=30)
            hp1, hp2 = 'lr', 'batch_size'
        elif case == 4:
            n1 = pow(10, np.linspace(-7, -2, num=30))
            n2 = np.linspace(8, 256, num=30)
            hp1, hp2 = 'lr', 'batch_size'
        elif case == 5:
            n1 = np.linspace(0.1, 1.0, num=30)
            n2 = np.linspace(32, 512, num=30)
            hp1, hp2 = 'dropout', 'fc_unit'

        config_dicts = []
        for item in itertools.product(n1, n2):
            config = {hp1: item[0], hp2: item[1], 'need_lc': False}
            config_dicts.append(config)
        T = config_dicts

        config_data = [[config[hp1], config[hp2]] for config in T]
        np.save('./data/grid_search_conf_%s_%s.npy' % (hp1, hp2), np.array(config_data))
        try:
            ret_val = self.run_in_parallel_easy(T, self.R)
            val_losses = [item['loss'] for item in ret_val]
            np.save('./data/grid_search_perf_%s_%s.npy' % (hp1, hp2), np.array(val_losses))
        except Exception as e:
            self.logger.error(str(e))
            np.save('./data/grid_search_perf_partial_%s_%s.npy' % (hp1, hp2), np.array(self.grid_search_perf))

    @BaseFacade.process_manage
    def run(self):
        try:
            self.logger.info('-'*50)
            self.logger.info("Grid Search starts")
            start_time = time.time()
            self.iterate()
            time_elapsed = (time.time() - start_time)/60
            self.logger.info("iteration took %.2f min." % time_elapsed)
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            self.remove_immediate_model()

