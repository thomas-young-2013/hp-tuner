import logging
import time
import dill
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def evaluate_func(params):
    objective_func, n_iteration, id, x = params
    objective_func = dill.loads(objective_func)
    start_time = time.time()
    return_val = objective_func(n_iteration, x)
    time_overhead = time.time() - start_time
    return return_val, time_overhead, id, x


class Runner(object):
    def __init__(self, objective_func, n_workers=1):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("logs/log_%s.txt" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

        self.objective_func = dill.dumps(objective_func)
        self.trial_statistics = []
        self.num_workers = n_workers
        self.pool = ProcessPoolExecutor(max_workers=n_workers)
        self.global_trial_counter = 0
        self.global_incumbent = 1e10
        self.global_incumbent_configuration = None

    def run_in_parallel(self, configurations, n_iteration, extra_info=None):
        n_configuration = len(configurations)
        batch_size = self.num_workers
        n_batch = n_configuration // batch_size + (1 if n_configuration%batch_size != 0 else 0)
        performance_result = []
        early_stops = []

        conf_list = configurations

        for i in range(n_batch):
            for config in conf_list[i*batch_size: (i+1)*batch_size]:
                self.trial_statistics.append(self.pool.submit(evaluate_func,
                    (self.objective_func, n_iteration, self.global_trial_counter, config)))
                self.global_trial_counter += 1

            # wait a batch of trials finish
            self.wait_tasks_finish()

            # get the evaluation statistics
            for trial in self.trial_statistics:
                assert(trial.done())
                return_info, time_taken, trail_id, config = trial.result()

                performance = return_info['loss']
                if performance < self.global_incumbent:
                    self.global_incumbent = performance
                    self.global_incumbent_configuration = config

                # TODO: old version => performance_result.append(performance)
                performance_result.append(return_info)
                early_stops.append(return_info.get('early_stop', False))

            self.trial_statistics.clear()
        return performance_result, early_stops

    def wait_tasks_finish(self):
        all_completed = False
        while not all_completed:
            all_completed = True
            for trial in self.trial_statistics:
                if not trial.done():
                    all_completed = False
                    time.sleep(0.1)
                    break

    def process_manage(func):
        def dec(*args):
            result = func(*args)
            args[0].garbage_collection()
            return result
        return dec

    def garbage_collection(self):
        self.pool.shutdown(wait=True)

    def get_incumbent(self):
        return self.global_incumbent, self.global_incumbent_configuration
