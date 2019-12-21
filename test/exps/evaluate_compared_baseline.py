import os
import sys
import argparse
import numpy as np
sys.path.append(os.getcwd())
from mfes.evaluate_function.hyperparameter_space_utils import get_benchmark_configspace
from mfes.facade.bohb import BOHB
from mfes.facade.hb import Hyperband
from mfes.facade.mfse import MFSE
from mfes.facade.mbhb import MBHB

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
                    choices=['fcnet', 'resnet', 'xgb'],
                    default='fcnet')
parser.add_argument('--baselines', type=str, default='hb,bohb,mfes')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--hb_iter', type=int, default=20000)
parser.add_argument('--runtime_limit', type=int, default=7200)
parser.add_argument('--rep_num', type=int, default=5)
args = parser.parse_args()

benchmark_id = args.benchmark
iter_num = args.hb_iter
maximal_iter = args.R
n_worker = args.n
runtime_limit = args.runtime_limit
baselines = args.baseline.split(',')
rep_num = args.rep_num
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))

# Generate random seeds.
np.random.seed(1)
seeds = np.random.randint(low=1, high=10000, size=rep_num)

# Load evaluation objective according to benchmark name.
if benchmark_id == 'fcnet':
    from mfes.evaluate_function.eval_fcnet_tf import train
elif benchmark_id == 'resnet':
    from mfes.evaluate_function.eval_resnet import train
elif benchmark_id == 'xgb':
    from mfes.evaluate_function.eval_xgb import train
else:
    raise ValueError('Unsupported Ojbective function: %s' % benchmark_id)


def evaluate_objective_function(cs):
    conf = cs.get_default_configuration().get_dictionary()
    conf['need_lc'] = False
    print(train(1, conf, None))


def evaluate_baseline(baseline_id, cs, id):
    _seed = seeds[id]
    if baseline_id == 'hb':
        optimizer = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, random_state=_seed)
    elif baseline_id == 'bohb':
        optimizer = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.3, n_workers=n_work, random_state=_seed)
    elif baseline_id == 'mbhb':
        optimizer = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, random_state=_seed)
    elif baseline_id == 'mfes':
        optimizer = MFSE(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, random_state=_seed)
    else:
        raise ValueError('Invalid baseline name: %s' % baseline_id)

    if benchmark_id == 'xgb':
        optimizer.restart_needed = True
    optimizer.runtime_limit = runtime_limit
    method_name = "%s-%s-%d-%d-%d" % (baseline_id, benchmark_id, id, runtime_limit, n_worker)
    optimizer.set_method_name(method_name)
    optimizer.run()
    print(optimizer.get_incumbent(5))
    return optimizer.get_incumbent(5)


if __name__ == "__main__":
    cs = get_benchmark_configspace(benchmark_id)
    for _id in range(rep_num):
        for _baseline in baselines:
            evaluate_baseline(_baseline, cs, _id)
