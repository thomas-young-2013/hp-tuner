import logging
import time
import dill
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
from mfes.utils.logging_utils import get_logger, setup_logger

plt.switch_backend('agg')


def evaluate_func(params):
    objective_func, n_iteration, id, x = params
    objective_func = dill.loads(objective_func)
    start_time = time.time()
    return_val = objective_func(n_iteration, x)
    time_overhead = time.time() - start_time
    return return_val, time_overhead, id, x


class BaseFacade(object):
    def __init__(self, objective_func, n_workers=1,
                 restart_needed=False, need_lc=False, method_name='Mth', log_directory='logs'):
        self.log_directory = log_directory
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        self.logger = self._get_logger("%s-%s" % (__class__.__name__, method_name))

        self.objective_func = dill.dumps(objective_func)
        self.trial_statistics = []
        self.num_workers = n_workers
        self.pool = ThreadPoolExecutor(max_workers=n_workers)
        self.recorder = []

        self.global_start_time = time.time()
        self.runtime_limit = None
        self._history = {"time_elapsed": [], "performance": [], "best_trial_id": [], "configuration": []}
        self.global_incumbent = 1e10
        self.global_incumbent_configuration = None
        self.global_trial_counter = 0
        self.restart_needed = restart_needed
        self.record_lc = need_lc
        self.method_name = method_name
        # evaluation metrics
        self.stage_id = 1
        self.stage_history = {'stage_id': [], 'performance': []}
        self.grid_search_perf = []

    def set_restart(self):
        self.restart_needed = True

    def set_method_name(self, name):
        self.method_name = name

    def add_stage_history(self, stage_id, performance):
        self.stage_history['stage_id'].append(stage_id)
        self.stage_history['performance'].append(performance)

    def add_history(self, time_elapsed, performance, trial_id, config):
        self._history['time_elapsed'].append(time_elapsed)
        self._history['performance'].append(performance)
        self._history['best_trial_id'].append(trial_id)
        self._history['configuration'].append(config)

    def run_in_parallel_easy(self, configuration_dicts, n_iteration):
        n_configuration = len(configuration_dicts)
        batch_size = self.num_workers
        n_batch = n_configuration // batch_size + (1 if n_configuration % batch_size != 0 else 0)
        performance_result = []
        self.logger.info('total batch number: %d' % n_batch)
        for i in range(n_batch):
            start_time = time.time()
            for config in configuration_dicts[i * batch_size: (i + 1) * batch_size]:
                self.trial_statistics.append(self.pool.submit(evaluate_func,
                                                              (self.objective_func, n_iteration,
                                                               self.global_trial_counter, config)))
                self.global_trial_counter += 1

            # wait a batch of trials finish
            self.wait_tasks_finish()

            # get the evaluation statistics
            for trial in self.trial_statistics:
                assert (trial.done())
                return_info, time_taken, trail_id, config = trial.result()

                performance = return_info['loss'][-1]
                if performance < self.global_incumbent:
                    self.global_incumbent = performance
                    self.global_incumbent_configuration = config

                performance_result.append(return_info)
                self.grid_search_perf.append(return_info['loss'])

            self.trial_statistics.clear()
            self.logger.info('evaluate %d-th batch: %.3f seconds' % (i, time.time() - start_time))
        return performance_result

    def run_in_parallel(self, configurations, n_iteration, extra_info=None):
        n_configuration = len(configurations)
        batch_size = self.num_workers
        n_batch = n_configuration // batch_size + (1 if n_configuration % batch_size != 0 else 0)
        performance_result = []
        early_stops = []

        # TODO: need systematic tests.
        # check configurations, whether it exists the same configs
        count_dict = dict()
        for i, config in enumerate(configurations):
            if config not in count_dict:
                count_dict[config] = 0
            count_dict[config] += 1

        # incorporate ref info.
        conf_list = []
        for index, config in enumerate(configurations):
            conf_dict = config.get_dictionary().copy()
            if count_dict[config] > 1:
                conf_dict['uid'] = count_dict[config]
                count_dict[config] -= 1

            if extra_info is not None:
                conf_dict['reference'] = extra_info[index]
            conf_dict['need_lc'] = self.record_lc
            conf_list.append(conf_dict)

        for i in range(n_batch):
            for config in conf_list[i * batch_size: (i + 1) * batch_size]:
                self.trial_statistics.append(self.pool.submit(evaluate_func,
                                                              (self.objective_func, n_iteration,
                                                               self.global_trial_counter, config)))

                self.global_trial_counter += 1

            # wait a batch of trials finish
            self.wait_tasks_finish()

            # get the evaluation statistics
            for trial in self.trial_statistics:
                assert (trial.done())

                return_info, time_taken, trail_id, config = trial.result()

                performance = return_info['loss']
                if performance < self.global_incumbent:
                    self.global_incumbent = performance
                    self.global_incumbent_configuration = config

                self.add_history(time.time() - self.global_start_time, self.global_incumbent, trail_id,
                                 self.global_incumbent_configuration)
                # TODO: old version => performance_result.append(performance)
                performance_result.append(return_info)
                early_stops.append(return_info.get('early_stop', False))
                self.recorder.append({'trial_id': trail_id, 'time_consumed': time_taken,
                                      'configuration': config, 'n_iteration': n_iteration})

            self.trial_statistics.clear()

        self.save_intemediate_statistics()
        if self.runtime_limit is not None and time.time() - self.global_start_time > self.runtime_limit:
            raise ValueError('Runtime budget meets!')
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

    def remove_immediate_model(self):
        data_dir = 'data/models'
        # filelist = [f for f in os.listdir(data_dir) if f.startswith("convnet") or f.startswith('checkpoint')]
        filelist = [f for f in os.listdir(data_dir)]
        for f in filelist:
            os.remove(os.path.join(data_dir, f))
        assert (len(os.listdir(data_dir)) == 0)

    def save_intemediate_statistics(self, save_stage=False):
        file_name = '%s.npy' % self.method_name
        x = np.array(self._history['time_elapsed'])
        y = np.array(self._history['performance'])
        np.save('data/%s' % file_name, np.array([x, y]))

        if save_stage:
            stage_file_name = 'stage_%s.npy' % self.method_name
            stage_x = np.array(self.stage_history['stage_id'])
            stage_y = np.array(self.stage_history['performance'])
            np.save('data/%s' % stage_file_name, np.array([stage_x, stage_y]))

        plt.plot(x, y)
        plt.xlabel('Time elapsed (sec)')
        plt.ylabel('Validation error')
        plt.savefig("data/%s.png" % self.method_name)

    def _get_logger(self, name):
        logger_name = 'mfes_%s' % name
        setup_logger(os.path.join(self.log_directory, '%s.log' % str(logger_name)), None)
        return get_logger(logger_name)
