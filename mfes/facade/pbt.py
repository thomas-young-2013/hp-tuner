import time
import random
import os
from mfes.facade.base_facade import BaseFacade
from mfes.utils.truncated_selection import select_top_worker
from mfes.config_space import get_random_neighborhood, sample_configurations


class Worker(object):
    def __init__(self, worker_id, obj_func, logger=None):
        self.w_id = worker_id
        self.obj_func = obj_func
        self.logger = logger

    def step(self, step_num, hp):
        args = hp.get_dictionary()
        result = self.obj_func(step_num, args, self.logger)
        result['worker_id'] = self.w_id
        return result

    @property
    def worker_id(self):
        return self.w_id


# TODO: different iter_gap's influence.
class BaseBPT(BaseFacade):
    def __init__(self, config_space, objective_func, n_population, iter_gap, iter_steps,
                 n_workers=1, iter_num=1, rand_int=123):
        BaseFacade.__init__(self, objective_func, n_workers=n_workers)
        self.config_space = config_space

        self.iter_gap = iter_gap
        self.iter_steps = iter_steps
        self.iter_num = iter_num
        self.n_population = n_population
        self.rand_int = rand_int

        self.workers = [Worker(wid, self.objective_func, self.logger) for wid in range(self.n_population)]

        self.counter = 0
        self.job_start_time = time.time()
        self.recorder= []
        self.incumbent_configs = []
        self.incumbent_obj = []

    def iterate(self):
        # sample n_population configurations.
        T = sample_configurations(self.config_space, self.n_population)
        step_num = 0
        result = []
        while step_num < self.iter_steps:
            step_num += self.iter_gap
            result = []
            for worker in self.workers:
                res = worker.step(step_num, T[worker.worker_id])
                result.append(res)
            T = self.hp_iterate(T, result)

        result_sorted = sorted(result, key=lambda x: x['loss'])
        self.incumbent_configs.append(T[result_sorted[0]['worker_id']])
        self.incumbent_obj.append(result_sorted[0]['loss'])

    def iterate_parallel(self):
        # sample n_population configurations.
        T = sample_configurations(self.config_space, self.n_population)
        self.logger.info('-'*20 + str([item.get_dictionary() for item in T]))
        step_num = 0
        result = []
        extra_info = None
        while step_num < self.iter_steps:
            self.logger.info('='*40 + ('start step: %d' % step_num) + '='*40)
            # step_num += self.iter_gap
            step_num += 1
            performance_result, early_stops = self.run_in_parallel(T, self.iter_gap, extra_info)
            result = []
            for i, item in enumerate(performance_result):
                result.append({'loss': item['loss'], 'worker_id': i, 'ref_id': item['ref_id']})
            T, extra_info = self.hp_iterate(T, result)
            self.logger.info('p update: ' + str([item.get_dictionary().values() for item in T]))

        result_sorted = sorted(result, key=lambda x: x['loss'])
        self.incumbent_configs.append(T[result_sorted[0]['worker_id']])
        self.incumbent_obj.append(result_sorted[0]['loss'])
    
    def get_neighbour_hp(self, T, worker_id):
        neighbours = get_random_neighborhood(T[worker_id], 10 * self.n_population, self.rand_int)
        for item in neighbours:
            if item not in T:
                return item
        # for _ in range(self.n_population):
        #     hp = random.choice(neighbours)
        #     if hp not in T:
        #         return hp
        return sample_configurations(self.config_space, 1)[0]

    def hp_iterate(self, T, result):
        ref_info = []
        for wid in range(self.n_population):
            worker_id = select_top_worker(wid, result)
            if worker_id is not None:
                T[wid] = self.get_neighbour_hp(T, worker_id)
                ref_info.append(result[worker_id]['ref_id'])
            else:
                ref_info.append(result[wid]['ref_id'])
        return T, ref_info

    @BaseFacade.process_manage
    def run(self):
        for iter in range(self.iter_num):
            self.logger.info('-'*50)
            self.logger.info("BPT algorithm: %d/%d iteration starts" % (iter, self.iter_num))
            self.iterate_parallel()
        for i, obj in enumerate(self.incumbent_obj):
            self.logger.info('%dth config: %s, obj: %f' % (i+1, str(self.incumbent_configs[i]), self.incumbent_obj[i]))
        self.remove_immediate_model()

