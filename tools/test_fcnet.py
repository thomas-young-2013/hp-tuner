import sys
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--iter', type=int, default=20000)
parser.add_argument('--iter_c', type=int, default=5)
parser.add_argument('--runtime_limit', type=int, default=7200)

args = parser.parse_args()

sys.path.append('/home/daim_gpu/sy/hp-tuner')
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
from hoist.facade.mfse import MFSE

iter_num = args.iter
maximal_iter = args.R
n_work = args.n
runtime_limit = args.runtime_limit
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def train_fcnet(cs):
    conf = cs.get_default_configuration().get_dictionary()
    conf['need_lc'] = False
    print(train(1, conf, None))


def test_bo(cs, id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.runtime_limit = runtime_limit
    method_name = "BO-fcnet-%d" % id
    bo.set_method_name(method_name)
    bo.run()
    bo.plot_statistics(method="BO-fcnet-%d" % id)
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_hb(cs, id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.runtime_limit = runtime_limit
    method_name = "HB-fcnet-%d" % id
    hyperband.set_method_name(method_name)
    hyperband.run()
    print(hyperband.get_incumbent(5))
    return hyperband.get_incumbent(5)


def test_bohb(cs, id):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.method_name = "BOHB-fcnet-%d" % id
    bohb.runtime_limit = runtime_limit
    bohb.run()
    bohb.plot_statistics(method="BOHB-fcnet-%d" % id)
    print(bohb.get_incumbent(5))
    return bohb.get_incumbent(5)


def test_hoist(cs, id, scale_mth=1):
    if scale_mth <= 6:
        weight = [0.2] * 5
    elif scale_mth == 7:
        weight = [0.0625, 0.125, 0.25, 0.5, 1.0]
    else:
        weight = None
    hoist = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, rho_delta=0.1, enable_rho=True,
                 scale_method=scale_mth, init_weight=weight)
    method_name = "HOIST-fcnet-%d-%d" % (scale_mth, id)
    hoist.method_name = method_name
    hoist.runtime_limit = runtime_limit
    hoist.run()

    hoist.plot_statistics(method=method_name)
    print(hoist.get_incumbent(5))
    weights = hoist.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return hoist.get_incumbent(5)


def test_mfse(cs, id):
    model = MFSE(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True)
    method_name = "MFSE_fcnet-%d" % id
    model.method_name = method_name
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    print(model.get_incumbent(5))
    weights = model.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return model.get_incumbent(5)


def test_mfse_average(cs, id):
    model = MFSE(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=False)
    method_name = "MFSE_AVERAGE_fcnet-%d" % id
    model.method_name = method_name
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    print(model.get_incumbent(5))
    weights = model.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return model.get_incumbent(5)


def test_mfse_single(cs, id):
    model = MFSE(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, multi_surrogate=False)
    method_name = "MFSE_SINGLE_fcnet-%d" % id
    model.method_name = method_name
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    print(model.get_incumbent(5))
    weights = model.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return model.get_incumbent(5)


def test_mbhb(cs, id):
    method_name = "MBHB-fcnet-%d" % id
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.runtime_limit = runtime_limit
    mbhb.set_method_name(method_name)
    mbhb.run()
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_boes(cs, id):
    method_name = "BOES-fcnet-%d" % id
    mbhb = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    mbhb.set_method_name(method_name)
    mbhb.runtime_limit = runtime_limit
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_vanilla_bo(cs, n_id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    method_name = "Vanilla-BO-fcnet-%d" % n_id
    bo.set_method_name(method_name)
    bo.runtime_limit = runtime_limit
    bo.run()
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_baseline_iid(cs, id):
    method_name = "BOSNG-fcnet-%d" % id
    mbhb = BaseIID(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.runtime_limit = runtime_limit
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def create_configspace():
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
    return cs


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    cs = create_configspace()
    # test_hb(cs, 1)
    # test_bohb(cs, 2)
    # test_vanilla_bo(cs, 3)
    # test_boes(cs, 4)
    test_bohb(cs, 11)
    test_bohb(cs, 12)
    test_bohb(cs, 13)
    test_bohb(cs, 14)
    test_bohb(cs, 15)
    test_mfse_single(cs, 21)
    test_mfse_single(cs, 22)
    test_mfse_single(cs, 23)
    test_mfse_single(cs, 24)
    test_mfse_single(cs, 25)
    test_mfse_average(cs, 31)
    test_mfse_average(cs, 32)
    test_mfse_average(cs, 33)
    test_mfse_average(cs, 34)
    test_mfse_average(cs, 35)
    test_mfse(cs, 41)
    test_mfse(cs, 42)
    test_mfse(cs, 43)
    test_mfse(cs, 44)
    test_mfse(cs, 45)
