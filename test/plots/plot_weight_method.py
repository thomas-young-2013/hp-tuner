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
parser.add_argument('--methods', type=str, default='rank_loss_softmax,rank_loss_single,rank_loss_prob,opt_based')
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--runtime_limit', type=int, default=7200)
parser.add_argument('--rep_num', type=int, default=5)
args = parser.parse_args()

benchmark_id = args.benchmark
n_worker = args.n
runtime_limit = args.runtime_limit
methods = args.methods.split(',')
rep_num = args.rep_num

color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']

if __name__ == "__main__":
    for idx,method in enumerate(methods):
        array_list = []
        for i in range(1, rep_num + 1):
            filename = "eval-w_%s-%s-%d-%d-%d.npy" % (method, benchmark_id, i, runtime_limit, n_worker)
            path = os.path.join("data", filename)
            array = np.load(path)
            interp_array = [np.interp(j, array[0], array[1]) for j in range(1, runtime_limit)]
            array_list.append(interp_array)
        array_list = np.average(np.array(array_list), axis=0)
        cut_start = 350
        x = np.linspace(cut_start, runtime_limit - 1, runtime_limit - 1 - cut_start)
        plt.xlabel("Time(s)")
        plt.ylabel("Valid Error")
        plt.plot(x, array_list[cut_start:], color=color_list[idx], label=method)

    plt.legend()
    plt.savefig("plot/Average.png")
