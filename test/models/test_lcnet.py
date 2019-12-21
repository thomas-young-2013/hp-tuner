import numpy as np
import matplotlib.pyplot as plt
import seaborn
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/hp-tuner')

from mfes.model.lcnet import LC_ES

seaborn.set_style(style='whitegrid')

plt.rc('text', usetex=True)
plt.rc('font', size=15.0, family='serif')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


def toy_example(t, a, b):
    return (10 + a * np.log(b * t)) / 10. + 10e-3 * np.random.rand()


current_palette = seaborn.color_palette("Paired", 10)
seaborn.set_palette(current_palette)

observed = 9
N = 200
n_epochs = 100
observed_t = int(n_epochs * (observed / 100.))

t_idx = np.arange(1, observed_t + 1) / n_epochs
t_grid = np.arange(1, n_epochs + 1) / n_epochs

configs = np.random.rand(N, 2)
learning_curves = [toy_example(t_grid, configs[i, 0], configs[i, 1]) for i in range(N)]

X_train = None
y_train = None
X_test = None
y_test = None

for i in range(20):

    x = np.repeat(configs[i, None, :], t_idx.shape[0], axis=0)
    x = np.concatenate((x, t_idx[:, None]), axis=1)

    x_test = np.concatenate((configs[i, None, :], np.array([[1]])), axis=1)

    lc = learning_curves[i][:observed_t]
    lc_test = np.array([learning_curves[i][-1]])

    if X_train is None:
        X_train = x
        y_train = lc
        X_test = x_test
        y_test = lc_test
    else:
        X_train = np.concatenate((X_train, x), 0)
        y_train = np.concatenate((y_train, lc), 0)
        X_test = np.concatenate((X_test, x_test), 0)
        y_test = np.concatenate((y_test, lc_test), 0)

    # plt.plot(t_idx * n_epochs, lc)
# plt.xlabel("Epochs", fontsize=20)
# plt.ylabel("Validation Accuracy", fontsize=20)
# plt.title("Training Data", fontsize=20)
# plt.xlim(1, n_epochs)
# plt.show()

lcnet = LC_ES()
print(y_train.shape)
lcnet.train(X_train, y_train)

test_config = 30
x = configs[test_config, None, :]
epochs = np.arange(1, n_epochs+1)
idx = epochs / n_epochs
x = np.repeat(x, idx.shape[0], axis=0)
x = np.concatenate((x, idx[:, None]), axis=1)
y_test = learning_curves[test_config].flatten()

m, v = lcnet.predict(x)
print(m.shape, v.shape)
print(m[-10:])
x = np.concatenate((configs[test_config, None, :], np.array([[1.0]])), axis=1)
print(x)
m, v = lcnet.predict(x)
print(m)
print(m.shape, v.shape)

# s = np.sqrt(v)
# plt.plot(epochs, y_test, color="black", label="True Learning Curve", linewidth=3)
#
#
# f, noise = lcnet.predict(x, return_individual_predictions=True)
#
# [plt.plot(epochs, fi, color="blue", alpha=0.08) for fi in f]
#
#
# plt.plot(epochs, m, color="red", label="LC-Net", linewidth=3)
# plt.legend(loc=4, fontsize=20)
# plt.xlabel("Epochs", fontsize=20)
# plt.ylabel("Validation Accuracy", fontsize=20)
# plt.xlim(1, n_epochs)
# plt.axvline(observed_t, linestyle="--", color="grey")
# plt.ylim(0, 1)
# plt.show()
