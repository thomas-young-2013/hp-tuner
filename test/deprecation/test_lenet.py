import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--n', type=int, default=10)
parser.add_argument('--iter', type=int, default=5)
parser.add_argument('--iter_c', type=int, default=5)

args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/running2/hp-tuner')

# Import ConfigSpace and different types of parameters
from mfes.config_space import ConfigurationSpace, sample_configurations
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from mfes.evaluate_function.eval_lenet_tf import train
from mfes.facade.bohb import BOHB
from mfes.facade.hb import Hyperband
from mfes.facade.pbt import BaseBPT
from mfes.facade.hoist import XFHB
from mfes.facade.batch_bo import SMAC
from mfes.facade.mbhb import MBHB
from mfes.facade.bo_es import SMAC_ES


iter_num = args.iter
maximal_iter = args.R
n_work = args.n
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def train_lenet(cs):
    print(train(1, cs.get_default_configuration().get_dictionary(), None))


def test_bo(cs):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.run()
    bo.plot_statistics(method="BO-lenet")
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_hb(cs):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.set_restart()
    hyperband.run()
    hyperband.plot_statistics(method="HB-lenet")
    print(hyperband.get_incumbent(5))
    return hyperband.get_incumbent(5)


def test_bohb(cs):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.set_restart()
    bohb.run()
    bohb.plot_statistics(method="BOHB-lenet")
    print(bohb.get_incumbent(5))
    return bohb.get_incumbent(5)


def test_pbt(cs):
    pbt = BaseBPT(cs, train, 10, 5, maximal_iter, n_workers=n_work, iter_num=iter_num)
    pbt.run()
    pbt.plot_statistics(method="PBT-lenet")


def test_xfhb(cs):
    xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work, info_type='Weighted',
                update_enable=True)
    xfhb.set_restart()
    xfhb.run()
    xfhb.plot_statistics(method="XFHB-lenet")
    print(xfhb.get_incumbent(5))
    return xfhb.get_incumbent(5)


def test_mbhb(cs):
    method_name = "MBHB-lenet"
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.set_restart()
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_boes(cs):
    method_name = "BOES-lenet"
    mbhb = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    # mbhb.set_restart()
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_methods_mean_variance(cs):
    iterations = args.iter_c
    for i in range(1, 1+iterations):
        weight = [1.0, 0.5, 0.25, 0.125, 0.125]
        bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
        bo.run()
        bo.plot_statistics(method="BO-lenet-%d" % i)

        hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
        hyperband.set_restart()
        hyperband.run()
        hyperband.plot_statistics(method="HB-lenet-%d" % i)

        bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
        bohb.set_restart()
        bohb.run()
        bohb.plot_statistics(method="BOHB-lenet-%d" % i)

        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted',
                    init_weight=weight, random_mode=False)
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="XFHB-lenet-disable_w-%d" % i)

        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted',
                    update_enable=True, update_delta=1, init_weight=weight, random_mode=False)
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="XFHB-lenet-update_w-%d" % i)


def test_update_rule(cs):
    iterations = args.iter_c
    for i in range(1, 1 + iterations):
        high_start_w = [1.0, 0.9, 0.72, 0.504, 0.3024, 0.1512]
        low_start_w = [1.0, 0.6, 0.5, 0.4, 0.4]
        weight = [1.0, 0.5, 0.25, 0.125, 0.125]
        zero_start = [0.0, 0.0, 0.0, 0.0, 0.0]
        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted',
                    update_enable=True, update_delta=1, init_weight=weight, random_mode=False)
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="XFHB-lenet-update_w_random_f-%d" % i)

        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted',
                    update_enable=True, update_delta=1, init_weight=weight, random_mode=True)
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="XFHB-lenet-update_w_random_t-%d" % i)

        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                    info_type='Weighted', init_weight=weight)
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="XFHB-lenet-not_update_w-%d" % i)

        hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
        hyperband.set_restart()
        hyperband.run()
        hyperband.plot_statistics(method="HB-lenet-new-%d" % i)

        bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
        bohb.set_restart()
        bohb.run()
        bohb.plot_statistics(method="BOHB-lenet-new-%d" % i)


if __name__ == "__main__":
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-4, 1e-2, default_value=1e-3, q=2e-4)
    n_layer1 = UniformIntegerHyperparameter("n_layer1", 16, 64, default_value=32, q=2)
    n_layer2 = UniformIntegerHyperparameter("n_layer2", 32, 96, default_value=64, q=2)
    batch_size = UniformIntegerHyperparameter("batch_size", 32, 128, default_value=64, q=8)
    dropout = UniformFloatHyperparameter("dropout", .3, .9, default_value=.5, q=.1)
    n_fc = UniformIntegerHyperparameter("fc_unit", 128, 512, default_value=256, q=64)
    cs.add_hyperparameters([learning_rate, n_layer1, n_layer2, batch_size, dropout, n_fc])

    # train_lenet(cs)
    # test_bo(cs)
    # test_bohb(cs)
    # test_hb(cs)
    # test_xfhb(cs)
    # test_methods_mean_variance(cs)
    # test_update_rule(cs)
    # test_mbhb(cs)
    test_boes(cs)
