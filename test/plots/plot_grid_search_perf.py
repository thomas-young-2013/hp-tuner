import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)
plt.rc('font', size=18.0, family='Times New Roman')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
pylab.rcParams['figure.figsize'] = (6.0, 5.0)

# plt.switch_backend('agg')
# dir = '/home/thomas/Desktop/paper-resource/data/assumption/intermediate_data/cifar_data/'
dir = '/home/thomas/Desktop/paper-resource/data/assumption/intermediate_data/'
# dir = '/home/thomas/Desktop/codes/hp-tuner/data/'


def plot():
    id = 'fc_unit_keep_prob'
    configs = np.load(dir+'grid_search_conf_%s.npy' % id)
    perfs = np.load(dir+'grid_search_perf_%s.npy' % id)

    abbr1 = [0.1, 0.325, 0.55, 0.775, 1.0]
    abbr2 = [1e-4, 48, 64, 80, 96]

    fig, axes = plt.subplots(nrows=1, ncols=4)
    for i in range(0, 4):
        # plt.cm.Greens
        ax = axes[i]
        ax = sns.heatmap(perfs[:, i].reshape((-1, 20)), ax=ax, cmap="YlGnBu", vmin=0, vmax=1.,
                         linewidths=.5, xticklabels=5, yticklabels=4)
        ax.set_xticklabels(abbr1)
        ax.set_yticklabels(abbr2)
        ax.set(xlabel=r"keep_prob \alpha", ylabel='lr')
    plt.show()
    # fig.savefig('./cmp.pdf', dpi=300)


def plot_single():
    case = 0
    if case == 1:
        n1 = np.linspace(0.1, 1.0, num=30)
        n2 = pow(10, np.linspace(-7, -2, num=30))
        hp1, hp2 = 'keep_prob', 'lr'
    elif case == 2:
        n1 = np.linspace(64, 768, num=30)
        n2 = np.linspace(0.1, 1.0, num=30)
        hp1, hp2 = 'fc_unit', 'keep_prob'
        bound_1 = [64, 768]
        bound_2 = [0.1, 1]
    elif case == 3:
        n1 = pow(10, np.linspace(-7, -2, num=30))
        n2 = np.linspace(16, 256, num=30)
        hp1, hp2 = 'lr', 'batch_size'
        bound_1 = [-7, -2]
        bound_2 = [16, 256]
    elif case == 4:
        n1 = pow(10, np.linspace(-7, -2, num=30))
        n2 = np.linspace(8, 256, num=30)
        hp1, hp2 = 'lr', 'batch_size'
        bound_1 = [-7, -2]
        bound_2 = [8, 256]
    elif case == 5:
        n1 = np.linspace(0.1, 1.0, num=30)
        n2 = np.linspace(32, 512, num=30)
        hp1, hp2 = 'dropout', 'fc_unit'
        bound_1 = [0.1, 1.0]
        bound_2 = [8, 512]
    elif case == 0:
        hp1, hp2 = 'keep_prob', 'lr'
        bound_2 = [0.1, 1.]
        bound_1 = [-7, -2]

    n_points = 30
    if case == 0:
        configs = np.load(dir + 'grid_search_conf.npy')
        perfs = np.load(dir + 'grid_search_perf.npy')
    else:
        configs = np.load(dir+'grid_search_conf_%s_%s.npy' % (hp1, hp2))
        perfs = np.load(dir+'grid_search_perf_%s_%s.npy' % (hp1, hp2))

    if case != 0:
        perfs -= 0.2
        abbr1 = np.linspace(bound_1[0], bound_1[1], num=6)
        abbr2 = np.linspace(bound_2[0], bound_2[1], num=6)
    else:
        abbr1 = np.linspace(bound_1[0], bound_1[1], num=6)
        abbr2 = np.linspace(bound_2[0], bound_2[1], num=6)
        abbr2[2] = 0.46
    print(configs[:90])

    for i in range(0, 4):
        fig, ax = plt.subplots()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)

        # To adjust.
        vmax = 1
        if case == 0:
            vmax = 0.15
        ax = sns.heatmap(perfs[:, i].reshape((-1, n_points)), ax=ax, cmap="YlGnBu", vmin=1e-3, vmax=vmax,
                         linewidths=.5)
        from matplotlib import ticker
        ax.set_xticklabels(abbr2)
        tick_locator = ticker.MaxNLocator(5)
        ax.xaxis.set_major_locator(tick_locator)
        # ax.set_xticklabels(abbr1)
        # ax.set_yticklabels(abbr2)
        ax.set_yticklabels(abbr1)
        ax.yaxis.set_major_locator(tick_locator)

        if case == 0:
            ax.set_xlabel("\\textbf{Keep Probability $\lambda$}", fontsize=24)
            ax.set_ylabel("\\textbf{Learning Rate $\\alpha$}", fontsize=24)
        else:
            ax.set(ylabel=hp1, xlabel=hp2)
        if case == 4:
            ax.set(ylabel='Learning Rate', xlabel="Batch Size")
        if case == 5:
            ax.set(ylabel='Dropout', xlabel="FC Units")
        # ax.set(ylabel='Learning Rate', xlabel="Batch Size")
        if case == 0:
            plt.subplots_adjust(top=0.98, right=1., left=0.11, bottom=0.12)
            fig.savefig(dir + 'intermediate_cmp_%d.pdf' % i, dpi=300)
        else:
            fig.savefig(dir+'intermediate_cmp_1_%s_%s_%d.pdf' % (hp1, hp2, i), dpi=300)


if __name__ == "__main__":
    plot_single()
