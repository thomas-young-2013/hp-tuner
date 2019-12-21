import random


def worker_is_bottom(id, population_sorted, num_selected_worker):
    bottom_worker = population_sorted[0:num_selected_worker]
    for worker in bottom_worker:
        if worker['worker_id'] == id:
            return True
    return False


# TODO: threshold need to decrease through iteration
def select_top_worker(id, population):
    population_sorted = sorted(population, key=lambda x: x['loss'])
    num_workers = len(population)
    num_selected_worker = int(num_workers * 0.2)
    if num_selected_worker < 1:
        num_selected_worker = 1
    if not worker_is_bottom(id, population_sorted, num_selected_worker):
        top_worker = population_sorted[0: num_selected_worker]
        return random.choice(top_worker)['worker_id']
    else:
        return None
