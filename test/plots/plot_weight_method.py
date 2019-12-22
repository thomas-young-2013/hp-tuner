import os
import sys
import pickle
import argparse
import numpy as np
sys.path.append(os.getcwd())
from mfes.evaluate_function.hyperparameter_space_utils import get_benchmark_configspace
from mfes.facade.mfse import MFSE

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
                    choices=['fcnet', 'resnet', 'xgb'],
                    default='fcnet')
parser.add_argument('--methods', type=str, default='softmax')
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
methods = args.methods.split(',')
rep_num = args.rep_num

# Generate random seeds.
np.random.seed(1)
seeds = np.random.randint(low=1, high=10000, size=rep_num)


def plot_curve():
    stats = dict()
    for method in methods:
        stats[method] = list()
        for id in range(rep_num):
            _seed = seeds[id]
            # Load data.
            method_name = "eval-w_%s-%s-%d-%d-%d" % (method, benchmark_id, id, runtime_limit, n_worker)
            with open('data/%s.npy' % method_name, 'rb') as f:
                data = pickle.load(f)



if __name__ == "__main__":
    plot_curve()
