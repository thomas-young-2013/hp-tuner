import numpy as np
from time import time


class RandomSearch(object):

    def __init__(self, config_space, objective_func, num_trails=50):
        self.config_space = config_space
        self.num_trials = num_trails
        self.objective_func = objective_func
        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        self.best_config = None

    def run(self):

        configs = self.config_space.sample_configuration(self.num_trials)

        for iter, config in enumerate(configs):
            self.counter += 1
            # print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
            #     self.counter, ctime(), self.best_loss, self.best_counter ))

            start_time = time()

            result = self.objective_func(config)

            assert (type(result) == dict)
            assert ('loss' in result)

            seconds = int(round(time() - start_time))
            # print("\n{} seconds.".format( seconds ))

            loss = result['loss']

            # keeping track of the best result so far (for display only)
            # could do it be checking results each time, but hey
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_counter = self.counter
                self.best_config = config

            result['counter'] = self.counter
            result['seconds'] = seconds
            result['params'] = config
            result['iterations'] = iter

            self.results.append(result)
        return self.best_config
