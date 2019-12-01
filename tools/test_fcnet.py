import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--iter_c', type=int, default=5)

args = parser.parse_args()

sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
sys.path.append('/home/daim/thomas/hp-tuner')

# Import ConfigSpace and different types of parameters
from hoist.config_space import ConfigurationSpace, sample_configurations
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from hoist.evaluate_function.eval_fcnet_tf import train
from hoist.facade.bohb import BOHB
from hoist.facade.hb import Hyperband
from hoist.facade.hoist import XFHB
from hoist.facade.batch_bo import SMAC
from hoist.facade.mbhb import MBHB
from hoist.facade.bo_es import SMAC_ES
from hoist.facade.baseline_iid import BaseIID


iter_num = args.iter
maximal_iter = args.R
n_work = args.n
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def train_fcnet(cs):
    conf = cs.get_default_configuration().get_dictionary()
    conf['need_lc'] = False
    print(train(1, conf, None))


def test_bo(cs, id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.run()
    bo.plot_statistics(method="BO-fcnet-%d" % id)
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_hb(cs, id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.runtime_limit = 19000
    method_name = "HB-fcnet-%d" % id
    hyperband.set_method_name(method_name)
    hyperband.run()
    print(hyperband.get_incumbent(5))
    return hyperband.get_incumbent(5)


def test_bohb(cs, id):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.run()
    bohb.plot_statistics(method="BOHB-fcnet-%d" % id)
    print(bohb.get_incumbent(5))
    return bohb.get_incumbent(5)


def test_hoist(cs, id, scale_mth=1):
    if scale_mth <= 6:
        weight = [0.2]*5
    elif scale_mth == 7:
        weight = [0.0625, 0.125, 0.25, 0.5, 1.0]
    else:
        weight = None
    hoist = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, rho_delta=0.1, enable_rho=True,
                 scale_method=scale_mth, init_weight=weight)
    method_name = "HOIST-fcnet-%d-%d" % (scale_mth, id)
    hoist.method_name = method_name
    hoist.runtime_limit = 18000
    hoist.run()

    hoist.plot_statistics(method=method_name)
    print(hoist.get_incumbent(5))
    weights = hoist.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return hoist.get_incumbent(5)


def test_mbhb(cs, id):
    method_name = "MBHB-fcnet-%d" % id
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.runtime_limit = 19000
    mbhb.set_method_name(method_name)
    mbhb.run()
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_boes(cs, id):
    method_name = "BOES-fcnet-%d" % id
    mbhb = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    mbhb.set_method_name(method_name)
    mbhb.runtime_limit = 19000
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_vanilla_bo(cs, n_id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    method_name = "Vanilla-BO-fcnet-%d" % n_id
    bo.set_method_name(method_name)
    bo.runtime_limit = 19000
    bo.run()
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_baseline_iid(cs, id):
    method_name = "BOSNG-fcnet-%d" % id
    mbhb = BaseIID(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


if __name__ == "__main__":
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-4, 1e-2, default_value=1e-3, q=2e-4)
    momentum = UniformFloatHyperparameter("momentum", 0., .5, default_value=0., q=.1)
    lr_decay = UniformFloatHyperparameter("lr_decay", .7, .99, default_value=9e-1, q=3e-2)

    n_layer1 = UniformIntegerHyperparameter("n_layer1", 32, 256, default_value=96, q=8)
    n_layer2 = UniformIntegerHyperparameter("n_layer2", 64, 256, default_value=128, q=8)
    batch_size = UniformIntegerHyperparameter("batch_size", 32, 128, default_value=64, q=8)
    dropout1 = UniformFloatHyperparameter("kb_1", .3, .9, default_value=.5, q=.1)
    dropout2 = UniformFloatHyperparameter("kb_2", .3, .9, default_value=.5, q=.1)
    kernel_regularizer = UniformFloatHyperparameter("k_reg", 1e-9, 1e-4, default_value=1e-6, q=5e-7, log=True)
    cs.add_hyperparameters([learning_rate, momentum, lr_decay, n_layer1, n_layer2, batch_size, dropout1, dropout2,
                            kernel_regularizer])

    test_hoist(cs, 0, scale_mth=6)
    test_hoist(cs, 1, scale_mth=6)
    test_hoist(cs, 2, scale_mth=6)
    test_hoist(cs, 3, scale_mth=6)
    test_hoist(cs, 4, scale_mth=6)
