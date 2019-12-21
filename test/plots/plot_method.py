import numpy as np
import matplotlib.pyplot as plt
import pylab

import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter

# %matplotlib inline
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

min_y = 1e10
task_mode = 5


def fx(x, func_list):
    perf_list = []
    for func in func_list:
        timestamp, perf = func
        last_p = 1.0
        for t, p in zip(timestamp, perf):
            if t > x:
                break
            last_p = p
        perf_list.append(last_p)
    return perf_list


def plot_mean_variance_timestamp():
    #            ['BO-rnn', 'BOHB-rnn',  'HB-rnn', 'HOIST-rnn-6', 'MBHB-rnn', 'Vanilla BO']
    color_list = ['purple', 'royalblue', 'green',  'red',         'brown',    'orange', 'yellowgreen']
    markers = ['s', '^', '2', 'o', 'v', 'p', '*']
    # markers = {0: 'o', 1: 's', 2: '^', 3: '2', 4: 'v', 5: 'x', 6: 'p', 7: 'd', 8: '<', 9: '>', 10: '1', 11: '.',
    #            12: '*'}

    # ['BOHB', 'Batch BO', 'FABOLAS', 'HB', 'HOIST', 'SMAC', 'TSE']
    color_list = ['royalblue', 'purple', 'brown', 'green', 'red', 'orange', 'yellowgreen']
    markers = ['^', 's', 'v', 'o', '*', 'p', '2']

    print(sorted(dict_data.keys()))
    print('='*100)
    color_dict, marker_dict = dict(), dict()
    for i, item in enumerate(sorted(dict_data.keys())):
        color_dict[item] = color_list[i]
        marker_dict[item] = markers[i]

    handles = []
    if task_mode == 5:
        method_list = ['SMAC', 'Batch BO', 'TSE', 'FABOLAS', 'HB', 'BOHB', 'HOIST']

    for method in method_list:
        function_list = dict_data[method]
        # x = np.linspace(1, np.log10(max_time), num=200)
        x = np.linspace(min_time, max_time, num=200)
        mean_t = []
        variance_t = []
        for i, stage in enumerate(x):
            # perf_list = fx(10**stage, function_list)
            perf_list = fx(stage, function_list)
            mean_t.append(np.mean(perf_list))
            variance_t.append(np.std(perf_list))

        # ax.errorbar(x, mean_t, yerr=variance_t, fmt="-", label=method)
        mean_t, variance_t = np.array(mean_t), np.array(variance_t)

        lw = 2
        ms = 4
        me = 10
        method_name = method
        if method.startswith('MBHB'):
            method_name = r'HB-LCNET'
            ax.plot(x, mean_t, lw=lw, label=method_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me)
        elif method.startswith('BO-'):
            method_name = r'Batch BO'
            ax.plot(x, mean_t, lw=lw, label=method_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me)
        elif method.startswith('SMAC'):
            method_name = method
            ax.plot(x, mean_t, lw=lw, label=method_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me)
        else:
            method_name = method.split('-')[0]
            ax.plot(x, mean_t, lw=lw, label=method_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me)

        line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                             markersize=ms, label=r'\textbf{%s}' % method_name)
        handles.append(line)
        # ax.fill_between(x, mean_t+variance_t, mean_t-variance_t, alpha=0.5)
        print(method, (mean_t[-1], variance_t[-1]))

    ax.set_ylim(0.149, 0.2)
    ax.set_xlim(min_time, max_time)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(max_time//10))
    legend = ax.legend(handles=handles, loc='best', ncol=2)
    ax.set_xlabel('\\textbf{wall clock time [s]}', fontsize=18)
    ax.set_ylabel('\\textbf{average validation error}', fontsize=18)

    plt.subplots_adjust(top=0.98, right=0.975, left=0.09, bottom=0.13)
    plt.savefig(dir + 'xgb_result.pdf')
    plt.show()


if __name__ == "__main__":
    plot_mean_variance_timestamp()
