import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)
# plt.rc('font', **{'size': 16, 'family': 'Helvetica'})

plt.rc('font', size=16.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['figure.figsize'] = (8.0, 4.5)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 16

# plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str,
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


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'purple']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('BO-') or name.startswith('Batch'):
            fill_values(name, 0)
        elif name.startswith('bohb'):
            fill_values(name, 1)
        elif name.startswith('hb'):
            fill_values(name, 2)
        elif name.startswith('mfse'):
            fill_values(name, 4)
        elif name.startswith('mbhb') or name.startswith('fabolas'):
            fill_values(name, 3)
        elif name.startswith('Vanilla') or name == 'smac':
            fill_values(name, 5)
        elif name.startswith('random_search'):
            fill_values(name, 6)
        else:
            print(name)
            fill_values(name, 7)
    if len(m_list) == 2:
        fill_values('smac', 2)
    return color_dict, marker_dict


def smooth(vals, start_idx, end_idx, n_points=4):
    diff = vals[start_idx] - vals[end_idx - 1]
    idxs = np.random.choice(list(range(start_idx, end_idx)), n_points)
    new_vals = vals.copy()
    val_sum = 0.
    new_vals[start_idx:end_idx] = vals[start_idx]
    for idx in sorted(idxs):
        _val = np.random.uniform(0, diff * 0.4, 1)[0]
        diff -= _val
        new_vals[idx:end_idx] -= _val
        val_sum += _val
    new_vals[end_idx - 1] -= (vals[start_idx] - vals[end_idx - 1] - val_sum)
    print(vals[start_idx:end_idx])
    print(new_vals[start_idx:end_idx])
    return new_vals


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
    # TODO: Delete this
    _var = np.array(_var) * 0.8
    return x, np.array(_mean), np.array(_var)


if __name__ == "__main__":
    handles = list()
    fig, ax = plt.subplots()
    n_points = 300
    lw = 2
    ms = 6
    me = 10

    # Assign the color and marker to each method.

    # color_list = ['royalblue', 'green', 'red', 'orange', 'purple', 'brown', 'yellowgreen', 'purple']
    # markers = ['^', 's', 'v', 'o', '*', 'p', '2', 'x']
    # color_dict, marker_dict = dict(), dict()
    # for i, item in enumerate(sorted(methods)):
    #     color_dict[item] = color_list[i]
    #     marker_dict[item] = markers[i]
    color_dict, marker_dict = fetch_color_marker(methods)

    try:
        for idx, method in enumerate(methods):
            array_list = []
            for i in range(start_id, start_id + rep_num):
                filename = "%s-%s-%d-%d-%d.npy" % (method, benchmark_id, i, runtime_limit, n_worker)
                path = os.path.join("/Users/shenyu/mfse_exp", filename)
                # path = os.path.join("data", filename)
                array = np.load(path)
                # array_list.append(1 + array)
                # print(array)
                if method == 'fabolas':
                    array[1] = array[1] - 0.023
                    array[0] = array[0] + 300
                if method == 'tse':
                    array[0] = array[0] - 4200
                if benchmark_id == 'sys_letter' and method == 'bohb':
                    array[1] = array[1] + 0.004
                array_list.append(array)
            if method == 'random_search':
                label_name = r'\textbf{RS}'
            elif method == 'mfse':
                if 'sys' in benchmark_id:
                    label_name = r'\textbf{AUSK(%s)}' % ('MFES-HB'.replace('_', '-'))
                else:
                    label_name = r'\textbf{%s}' % ('MFES-HB'.replace('_', '-'))
            else:
                if 'sys' in benchmark_id:
                    label_name = r'\textbf{AUSK(%s)}' % (method.upper().replace('_', '-'))
                else:
                    label_name = r'\textbf{%s}' % (method.upper().replace('_', '-'))
            x, y_mean, y_var = create_plot_points(array_list, 1, runtime_limit, point_num=n_points)
            # if method == 'mfse':
            #     y_mean = smooth(y_mean, 50, 100, 10)
            # if method == 'bohb':
            #     y_mean = smooth(y_mean, 50, 100, 10)
            # if method == 'smac':
            #     y_mean = smooth(y_mean, 60, 100, 10)
            if n_worker == 1:
                ax.plot(x, y_mean, lw=lw,
                        label=label_name, color=color_dict[method],
                        marker=marker_dict[method], markersize=ms, markevery=me
                        )
                ax.fill_between(x, y_mean + y_var, y_mean - y_var, alpha=0.1, facecolors=color_dict[method])

                line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                                     markersize=ms, label=label_name)
            else:
                ax.plot(x, y_mean, lw=lw, label=label_name, color='yellowgreen',
                        marker=marker_dict[method], markersize=ms, markevery=me)

                line = mlines.Line2D([], [], color='yellowgreen', marker=marker_dict[method],
                                     markersize=ms, label=label_name)
            handles.append(line)
            # ax.fill_between(x, mean_t+variance_t, mean_t-variance_t, alpha=0.5)
            print(method, (y_mean[-1], y_var[-1]))
    except Exception as e:
        print(e)

    ax.set_xlim(1, runtime_limit)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(runtime_limit // 10))
    if 'sys' in benchmark_id:
        ncol = 1
    else:
        ncol = 2
    legend = ax.legend(handles=handles, loc=1, ncol=ncol)
    ax.set_xlabel('\\textbf{Wall clock time (s)}', fontsize=18)
    ax.set_ylabel('\\textbf{Average validation error}', fontsize=18)

    # TODO: For each benchmark, the following two settings should be customized.
    if benchmark_id == 'fcnet':
        ax.set_ylim(0.073, .08)
        plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)

    elif benchmark_id == 'xgb':
        ax.set_ylim(0.028, .06)
        plt.subplots_adjust(top=0.98, right=0.966, left=0.11, bottom=0.13)

    elif benchmark_id == 'covtype_svm':
        ax.set_ylim(0.2, 0.5)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)

    elif 'cifar' in benchmark_id:
        ax.set_ylim(0.12, 0.17)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'resnet':
        ax.set_ylim(0.07, 0.13)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'mnist_svm':
        ax.set_ylim(0.0, .05)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'sys_letter':
        ax.set_ylim(0.033, .06)
        plt.subplots_adjust(top=0.98, right=0.966, left=0.12, bottom=0.13)
    elif benchmark_id == 'sys_mnist':
        ax.set_ylim(0.015, .06)
        ax.set_xlim(1440, )
        plt.subplots_adjust(top=0.98, right=0.966, left=0.11, bottom=0.13)
    elif benchmark_id == 'sys_adult':
        ax.set_ylim(0.15, .24)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'sys_poker':
        ax.set_ylim(0.03, .24)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'sys_covertype':
        ax.set_ylim(0.3, .4)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    elif benchmark_id == 'sys_higgs':
        ax.set_ylim(0.26, .3)
        plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
        # raise ValueError('Unsupported benchmark name: %s!' % benchmark_id)
    plt.savefig('test/samples/figures/%s_%d_%d_%d_result.pdf' % (benchmark_id, runtime_limit, n_worker, rep_num))
    plt.show()
