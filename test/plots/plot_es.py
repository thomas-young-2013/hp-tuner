import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab
from scipy.interpolate import spline

# sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)
plt.rc('font', size=18.0, family='Times New Roman')
plt.rcParams['figure.figsize'] = (8.0, 3.0)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# pylab.rcParams['figure.figsize'] = (8.0, 5.0)

# plt.switch_backend('agg')


def plot_single():
    data = np.load('./data/es_data.npy')
    print(data.shape)
    max_x = 450
    x = np.linspace(1, max_x, num=max_x)
    # 3rd: best
    # secondly best: 5-th, 7-th.
    color_list = ['mediumpurple', 'rosybrown', 'green', 'red', 'burlywood', 'cadetblue', 'maroon',
    'royalblue', 'hotpink', 'royalblue']
    index = 0
    for item in data:
        if len(item) < 13:
            continue
        item = (np.array(item)-0.3)*3
        max_x = len(item)
        xnew = np.linspace(1, max_x, max_x*4)
        y_smooth = spline(x[:max_x], item, xnew)
        plt.plot(xnew, y_smooth, color=color_list[index], lw=2)
        index += 1
        # plt.plot(x[:len(item)], item)
    print(index)
    plt.xlabel('\\textbf{training resource [epochs]}')
    plt.ylabel('\\textbf{validation error}')
    plt.axvline(9*4, linestyle="--", color="cadetblue", lw=2)
    plt.axvline(27*4, linestyle="--", color="cadetblue", lw=2)
    plt.axvline(81*4, linestyle="--", color="cadetblue", lw=2)

    plt.annotate('\\textbf{1st early stop: 9 low-fidelity evaluation data $D_1$}', xy=(36, 0.3), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=15,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.annotate('\\textbf{2nd early stop: 3 low-fidelity evaluation data $D_2$}', xy=(27*4, 0.2), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=15,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.annotate('\\textbf{1 high-fidelity evaluation data $D_3$}', xy=(81*4, 0.1), xycoords='data',
                 xytext=(-90, +30), textcoords='offset points', fontsize=15,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.legend()
    plt.xlim(1, 450)
    plt.ylim(0.0, 0.45)
    plt.subplots_adjust(top=0.98, right=0.97, left=0.09, bottom=0.21)
    plt.savefig('hb_es.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_single()
