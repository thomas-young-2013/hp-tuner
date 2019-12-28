import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)
plt.rc('font', size=12.0, family='Times New Roman')
plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'black'
plt.rc('legend', **{'fontsize': 12})

# plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
                    choices=['fcnet', 'resnet', 'xgb'],
                    default='fcnet')
parser.add_argument('--methods', type=str, default='hb,bohb,mbhb,boes,mfse,smac,random_search')
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--runtime_limit', type=int, default=7200)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=5)
args = parser.parse_args()

benchmark_id = args.benchmark
n_worker = args.n
runtime_limit = args.runtime_limit
methods = args.methods.split(',')
rep_num = args.rep_num
start_id = args.start_id
color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']


def create_point(x, stats):
    perf_list = []
    for func in stats:
        timestamp, perf = func
        last_p = 1.0
        for t, p in zip(timestamp, perf):
            if t > x:
                break
            last_p = p
        perf_list.append(last_p)
    return perf_list


def create_plot_points(data, start_time, end_time, point_num=500):
    x = np.linspace(start_time, end_time, num=point_num)
    _mean, _var = list(), list()
    for i, stage in enumerate(x):
        perf_list = create_point(stage, data)
        _mean.append(np.mean(perf_list))
        _var.append(np.std(perf_list))
    # Used to plot errorbar.
    return x, np.array(_mean), np.array(_var)


if __name__ == "__main__":
    handles = list()
    fig, ax = plt.subplots()
    n_points = 300
    lw = 2
    ms = 4
    me = 10

    # Assign the color and marker to each method.
    color_list = ['royalblue', 'purple', 'brown', 'green', 'red', 'orange', 'yellowgreen', 'purple']
    markers = ['^', 's', 'v', 'o', '*', 'p', '2', 'x']
    color_dict, marker_dict = dict(), dict()
    for i, item in enumerate(sorted(methods)):
        color_dict[item] = color_list[i]
        marker_dict[item] = markers[i]

    for idx, method in enumerate(methods):
        array_list = []
        for i in range(start_id, start_id + rep_num):
            filename = "%s-%s-%d-%d-%d.npy" % (method, benchmark_id, i, runtime_limit, n_worker)
            path = os.path.join("data", filename)
            array = np.load(path)
            array_list.append(array)
        label_name = r'\textbf{%s}' % method.replace('_', '-')
        x, y_mean, y_var = create_plot_points(array_list, 1, runtime_limit, point_num=n_points)
        ax.plot(x, y_mean, lw=lw, label=label_name, color=color_dict[method],
                marker=marker_dict[method], markersize=ms, markevery=me)

        line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                             markersize=ms, label=label_name)
        handles.append(line)
        # ax.fill_between(x, mean_t+variance_t, mean_t-variance_t, alpha=0.5)
        print(method, (y_mean[-1], y_var[-1]))

    ax.set_xlim(1, runtime_limit)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(runtime_limit // 10))
    legend = ax.legend(handles=handles, loc='best', ncol=2)
    ax.set_xlabel('\\textbf{wall clock time [s]}', fontsize=18)
    ax.set_ylabel('\\textbf{average validation error}', fontsize=18)

    # TODO: For each benchmark, the following two settings should be customized.
    if benchmark_id == 'fcnet':
        ax.set_ylim(0.06, .14)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'xgb':
        ax.set_ylim(0.06, .14)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    else:
        raise ValueError('Unsupported benchmark name: %s!' % benchmark_id)
    plt.savefig('test/samples/png/%s_%d_%d_%d_result.pdf' % (benchmark_id, runtime_limit, n_worker, rep_num))
    plt.show()
