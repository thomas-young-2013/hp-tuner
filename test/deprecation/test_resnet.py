import os
import sys
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--iter_c', type=int, default=5)
parser.add_argument('--b', type=int, default=50000)

args = parser.parse_args()

if args.mode == 'server':
    sys.path.append('/root/sy/hp-tuner')
elif args.mode == 'gpu':
    sys.path.append('/home/daim_gpu/sy/hp-tuner')
else:
    raise ValueError('Invalid mode!')

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter
from mfes.facade.bohb import BOHB
from mfes.facade.hb import Hyperband
from mfes.facade.hoist import XFHB
from mfes.facade.batch_bo import SMAC
from mfes.facade.mbhb import MBHB
from mfes.facade.bo_es import SMAC_ES
from mfes.facade.baseline_iid import BaseIID
from mfes.evaluate_function.eval_resnet import train

iter_num = args.iter
maximal_iter = args.R
n_work = args.n
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def create_hyperspace():
    cs = ConfigurationSpace()
    padding_size = CategoricalHyperparameter('padding_size', [1, 2, 3], default_value=2)
    # batch_size = CategoricalHyperparameter('train_batch_size', [256])
    batch_size = UniformIntegerHyperparameter("train_batch_size", 32, 256, default_value=64, q=8)
    init_lr = UniformFloatHyperparameter('init_lr', lower=1e-3, upper=0.3, default_value=0.1, log=True)
    lr_decay_factor = UniformFloatHyperparameter('lr_decay_factor', lower=0.01, upper=0.2, default_value=0.1, log=True)
    weight_decay = UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, default_value=0.0002, log=True)
    momentum = UniformFloatHyperparameter("momentum", 0.5, .99, default_value=0.9)
    cs.add_hyperparameters([padding_size, batch_size, init_lr, lr_decay_factor, weight_decay, momentum])
    return cs


def train_resnet(cs):
    conf = cs.get_default_configuration().get_dictionary()
    conf['need_lc'] = False
    print(train(10, conf, None))


def save_result(file, result):
    folder = 'data/hpo_result/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + file + '.data', 'wb') as f:
        pickle.dump(result, f)


def test_bo(cs, id):
    method_id = "BO-resnet-%d" % id
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.restart_needed = True
    bo.runtime_limit = args.b
    bo.run()
    bo.plot_statistics(method=method_id)
    result = bo.get_incumbent(5)
    save_result(method_id, result)
    print(result)
    return result


def test_hb(cs, id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.runtime_limit = args.b
    hyperband.restart_needed = True
    method_id = "HB-resnet-%d" % id
    hyperband.set_method_name(method_id)
    hyperband.run()
    result = hyperband.get_incumbent(5)
    save_result(method_id, result)
    return result


def test_bohb(cs, id):
    method_id = "BOHB-resnet-%d" % id
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.restart_needed = True
    bohb.runtime_limit = args.b
    bohb.run()
    bohb.plot_statistics(method=method_id)
    result = bohb.get_incumbent(5)
    save_result(method_id, result)
    return result


def test_mbhb(cs, n_id):
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    method_id = "MBHB-resnet-%d" % n_id
    mbhb.set_method_name(method_id)
    mbhb.runtime_limit = args.b
    mbhb.restart_needed = True
    mbhb.run()
    result = mbhb.get_incumbent(5)
    save_result(method_id, result)
    return result


def test_boes(cs, n_id):
    method_id = "BOES-resnet-%d" % n_id
    boes = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    boes.set_method_name(method_id)
    boes.runtime_limit = args.b
    boes.restart_needed = True
    boes.run()
    result = boes.get_incumbent(5)
    save_result(method_id, result)
    return result


def test_vanilla_bo(cs, n_id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    method_name = "Vanilla-BO-resnet-%d" % n_id
    bo.set_method_name(method_name)
    bo.runtime_limit = args.b
    bo.run()
    result = bo.get_incumbent(5)
    print(result)
    save_result(method_name, result)
    return result


def test_baseline_iid(cs, id):
    method_name = "BOSNG-resnet-%d" % id
    mbhb = BaseIID(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.runtime_limit = args.b
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    result = mbhb.get_incumbent(5)
    save_result(method_name, result)
    return mbhb.get_incumbent(5)


def test_hoist(cs, id, scale_mth=6):
    if scale_mth == 6:
        weight = [0.2] * 5
    elif scale_mth == 7:
        weight = [0.0625, 0.125, 0.25, 0.5, 1.0]
    else:
        weight = None
    # mfes = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
    #              update_enable=True, rho_delta=0.1, enable_rho=True,
    #              scale_method=scale_mth, init_weight=weight)
    hoist = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, rho_delta=0.1, enable_rho=False, init_rho=0.5,
                 scale_method=scale_mth, init_weight=weight)
    hoist.restart_needed = True
    hoist.runtime_limit = args.b
    method_id = "HOIST-resnet-%d-%d" % (scale_mth, id)
    hoist.set_method_name(method_id)
    hoist.run()
    result = hoist.get_incumbent(5)
    save_result(method_id, result)
    return result


if __name__ == '__main__':
    cs = create_hyperspace()
    test_hb(cs, 0)
    test_hb(cs, 1)
    test_bo(cs, 0)
    test_bo(cs, 1)
    # train_resnet(cs)
