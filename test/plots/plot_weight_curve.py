import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sys

plt.switch_backend('agg')
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
                    choices=['fcnet', 'resnet', 'xgb'],
                    default='fcnet')
parser.add_argument('--method', type=str, default='rank_loss_softmax',
                    choices=['rank_loss_softmax','rank_loss_single','rank_loss_prob','opt_based'])
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--runtime_limit', type=int, default=7200)
parser.add_argument('--id', type=int, default=2)
args = parser.parse_args()

benchmark_id = args.benchmark
n_worker = args.n
runtime_limit = args.runtime_limit
method = args.method
idx = args.id


if __name__ == "__main__":
    filename = "eval-w_%s-%s-%d-%d-%d_weights_eval-w_%s-%s-%d-%d-%d.npy" % (
        method, benchmark_id, idx, runtime_limit, n_worker,
        method, benchmark_id, idx, runtime_limit, n_worker)
    path = os.path.join("data", filename)
    weight_array = np.load(path)
    new_array = np.transpose(weight_array)
    plt.xlabel('iteration')
    plt.ylabel('weight')
    plt.xticks(range(1, int(1 + new_array.shape[1] / 5) + 1, 1))
    x = np.linspace(1, 1 + new_array.shape[1] / 5, new_array.shape[1])
    for i in range(5):
        plt.plot(x, new_array[i], label='sur%d' % (i + 1))

    plt.legend()
    plt.savefig("plot/weight_%s_%s.png" % (method, benchmark_id))
