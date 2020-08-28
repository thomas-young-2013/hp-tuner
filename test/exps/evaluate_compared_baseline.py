import os
import sys
import argparse
import numpy as np
import pickle as pkl

sys.path.append(os.getcwd())
from mfes.evaluate_function.hyperparameter_space_utils import get_benchmark_configspace
from mfes.facade.bohb import BOHB
from mfes.facade.hb import Hyperband
from mfes.facade.mfse import MFSE
from mfes.facade.mbhb import MBHB
from mfes.facade.bo_es import SMAC_ES
from mfes.facade.batch_bo import SMAC
from mfes.facade.random_search import RandomSearch
from mfes.facade.fabolas import FABOLAS
from mfes.facade.tse import TSE

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
                    choices=['fcnet', 'cifar', 'svhn', 'covtype', 'covtype_svm', 'higgs', 'convnet'],
                    default='fcnet')
parser.add_argument('--baseline', type=str, default='hb,bohb,mfse')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--hb_iter', type=int, default=50000)
parser.add_argument('--runtime_limit', type=int, default=18000)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--cuda_device', type=str, default='2')
args = parser.parse_args()

benchmark_id = args.benchmark
iter_num = args.hb_iter
maximal_iter = args.R
n_worker = args.n
runtime_limit = args.runtime_limit
baselines = args.baseline.split(',')
rep_num = args.rep_num
start_id = args.start_id
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_worker))

# Generate random seeds.
np.random.seed(1)
seeds = np.random.randint(low=1, high=10000, size=start_id + rep_num)

# Load evaluation objective according to benchmark name.
if benchmark_id == 'fcnet':
    from mfes.evaluate_function.eval_fcnet_tf import train
elif benchmark_id == 'cifar':
    from mfes.evaluate_function.eval_cifar import train
elif benchmark_id == 'svhn':
    from mfes.evaluate_function.eval_svhn import train
elif benchmark_id == 'covtype':
    from mfes.evaluate_function.eval_covtype import train
elif benchmark_id == 'covtype_svm':
    from mfes.evaluate_function.eval_covtype_svm import train
elif benchmark_id == 'higgs':
    from mfes.evaluate_function.eval_higgs import train
elif benchmark_id == 'resnet_cifar100':
    from mfes.evaluate_function.eval_resnet_cifar100 import train
elif benchmark_id == 'convnet':
    from mfes.evaluate_function.eval_convnet_tf import train
else:
    raise ValueError('Unsupported Ojbective function: %s' % benchmark_id)


def evaluate_objective_function(baseline_id, id):
    method_name = "%s-%s-%d-%d-%d" % (baseline_id, benchmark_id, id, runtime_limit, n_worker)
    with open('data/config_%s.npy' % method_name, 'rb') as f:
        conf = pkl.load(f)
    from mfes.evaluate_function.eval_cifar import eval
    result = eval(200, conf)
    with open('data/result_%s.pkl' % method_name, 'wb') as f:
        pkl.dump(result, f)


def evaluate_baseline(baseline_id, cs, id):
    _seed = seeds[id]
    method_name = "%s-%s-%d-%d-%d" % (baseline_id, benchmark_id, id, runtime_limit, n_worker)
    if baseline_id == 'hb':
        optimizer = Hyperband(cs, train, maximal_iter, num_iter=iter_num,
                              n_workers=n_worker, random_state=_seed, method_id=method_name)
    elif baseline_id == 'bohb':
        optimizer = BOHB(cs, train, maximal_iter, num_iter=iter_num,
                         p=0.3, n_workers=n_worker, random_state=_seed, method_id=method_name)
    elif baseline_id == 'mbhb':
        optimizer = MBHB(cs, train, maximal_iter, num_iter=iter_num,
                         n_workers=n_worker, random_state=_seed, method_id=method_name)
    elif baseline_id == 'mfse':
        optimizer = MFSE(cs, train, maximal_iter, num_iter=iter_num, weight_method='rank_loss_prob',
                         n_workers=n_worker, random_state=_seed, method_id=method_name)
    elif baseline_id == 'smac':
        optimizer = SMAC(cs, train, maximal_iter, num_iter=iter_num,
                         n_workers=1, random_state=_seed, method_id=method_name)
    elif baseline_id == 'batch_bo':
        optimizer = SMAC(cs, train, maximal_iter, num_iter=iter_num,
                         n_workers=n_worker, random_state=_seed, method_id=method_name)
    elif baseline_id == 'boes':
        optimizer = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num,
                            n_workers=n_worker, random_state=_seed, method_id=method_name)
    elif baseline_id == 'random_search':
        optimizer = RandomSearch(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_worker,
                                 random_state=_seed, method_id=method_name)
    elif baseline_id == 'fabolas':  # only for xgb R=27
        optimizer = FABOLAS(method_id=method_name)
    elif baseline_id == 'tse':
        optimizer = TSE(n_workers=n_worker, method_id=method_name)
    else:
        raise ValueError('Invalid baseline name: %s' % baseline_id)

    if benchmark_id == 'xgb':
        optimizer.restart_needed = True
    optimizer.runtime_limit = runtime_limit

    optimizer.run()
    print(optimizer.get_incumbent(5))
    return optimizer.get_incumbent(5)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
    cs = get_benchmark_configspace(benchmark_id)
    for _id in range(start_id, start_id + rep_num):
        for _baseline in baselines:
            evaluate_baseline(_baseline, cs, _id)
            # evaluate_objective_function(_baseline, _id)
