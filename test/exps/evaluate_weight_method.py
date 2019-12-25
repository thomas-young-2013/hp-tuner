import os
import sys
import argparse
import numpy as np

sys.path.append(os.getcwd())
from mfes.evaluate_function.hyperparameter_space_utils import get_benchmark_configspace
from mfes.facade.mfse import MFSE

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
                    choices=['fcnet', 'resnet', 'xgb'],
                    default='fcnet')
parser.add_argument('--methods', type=str, default='rank_loss_p_norm-2,rank_loss_softmax,rank_loss_single,rank_loss_prob,opt_based')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--hb_iter', type=int, default=20000)
parser.add_argument('--runtime_limit', type=int, default=7200)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--cuda_device', type=str, default='0')
args = parser.parse_args()

benchmark_id = args.benchmark
iter_num = args.hb_iter
maximal_iter = args.R
n_worker = args.n
runtime_limit = args.runtime_limit
methods = args.methods.split(',')
rep_num = args.rep_num
start_id = args.start_id
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_worker))

# Generate random seeds.
np.random.seed(1)
seeds = np.random.randint(low=1, high=10000, size=start_id+rep_num)

# Load evaluation objective according to benchmark name.
if benchmark_id == 'fcnet':
    from mfes.evaluate_function.eval_fcnet_tf import train
elif benchmark_id == 'resnet':
    from mfes.evaluate_function.eval_resnet import train
elif benchmark_id == 'xgb':
    from mfes.evaluate_function.eval_xgb import train
else:
    raise ValueError('Unsupported Ojbective function: %s' % benchmark_id)


def evaluate_weight_learning(method, cs, id):
    _seed = seeds[id]
    method_name = "eval-w_%s-%s-%d-%d-%d" % (method, benchmark_id, id, runtime_limit, n_worker)

    power_num = 0
    if method.startswith('rank_loss_p_norm'):
        method, power_num = method.split('-')
        power_num = int(power_num)
        assert power_num in [1, 2, 3, 4, 5]

    optimizer = MFSE(cs, train, maximal_iter,
                     weight_method=method,
                     num_iter=iter_num,
                     n_workers=n_worker,
                     random_state=_seed,
                     method_id=method_name,
                     power_num=power_num)

    if benchmark_id == 'xgb':
        optimizer.restart_needed = True

    optimizer.runtime_limit = runtime_limit

    optimizer.run()
    print(optimizer.get_incumbent(5))
    return optimizer.get_incumbent(5)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    cs = get_benchmark_configspace(benchmark_id)
    for idx in range(start_id, start_id + rep_num):
        for _method in methods:
            evaluate_weight_learning(_method, cs, idx)
