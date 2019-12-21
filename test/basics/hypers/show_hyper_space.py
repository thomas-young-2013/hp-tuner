import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from mfes.evaluate_function.eval_lenet_tf import train
from mfes.utils.runner import Runner

data_file = 'data/exp_data.npy'


def collect(x1_bound, x2_bound, gap, iter_num=10, train_needed=False, n_worker=10):
    x1s = range(x1_bound[0], x1_bound[1]+1, gap)
    x2s = range(x2_bound[0], x2_bound[1]+1, gap)

    X, Y = np.meshgrid(x1s, x2s)
    shape = X.shape
    print(X.shape, Y.shape)
    if train_needed:
        config_list = []
        for item in zip(X.flat, Y.flat):
            config = {'n_layer1': item[0], 'n_layer2': item[1]}
            config_list.append(config)

        print('Total number of Configs: %d' % len(config_list))
        runner = Runner(train, n_workers=n_worker)
        res, _ = runner.run_in_parallel(config_list, iter_num)

        z = []
        for item in res:
            z.append(item['loss'])
        Z = np.array(z).reshape(shape)
        np.save(data_file, Z)
    else:
        Z = np.load(data_file)
        Z[Z > 10] = 10
        print(np.argmin(Z.flat))
        data = list(zip(X.flat, Y.flat))
        print(data[564])
    plt.contourf(X, Y, Z, cmap=plt.cm.hot)
    C = plt.contour(X, Y, Z)
    plt.clabel(C, inline=True, fontsize=10)
    cb = plt.colorbar()
    cb.set_label('loss')
    plt.savefig('./res_1.pdf')
    plt.show()


if __name__ == '__main__':
    x1_bound = (8, 64)
    x2_bound = (32, 96)
    gap = 2
    collect(x1_bound, x2_bound, gap, iter_num=10, train_needed=False, n_worker=10)
