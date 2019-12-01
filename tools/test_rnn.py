import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--iter_c', type=int, default=5)

args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/running/hp-tuner')

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition
from hoist.config_space import ConfigurationSpace
from hoist.evaluate_function.eval_lstm_tf import train
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


def create_hyperspace():
    cs = ConfigurationSpace()

    # training hyperparameters.
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 5e-1, default_value=1e-3, q=3e-5, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", 64, 196, q=16, default_value=128)
    lstm_units = UniformIntegerHyperparameter("lstm_units", 32, 128, default_value=48, q=8)
    keep_prob = UniformFloatHyperparameter("keep_prob", .1, 1., default_value=.75, q=.05)
    cs.add_hyperparameters([learning_rate, batch_size, lstm_units, keep_prob])

    # configuration = cs.get_default_configuration()
    # print(configuration.configuration_space)
    # print(configuration.get_dictionary())
    return cs


def train_rnn(cs):
    print(cs.get_default_configuration().get_dictionary())
    print(train(maximal_iter, cs.get_default_configuration().get_dictionary(), None))


def test_bo(cs, id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.run()
    bo.plot_statistics(method="BO-rnn-%d" % id)
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_hb(cs, id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.run()
    hyperband.plot_statistics(method="HB-rnn-%d" % id)
    print(hyperband.get_incumbent(5))
    return hyperband.get_incumbent(5)


def test_bohb(cs, id):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.run()
    bohb.plot_statistics(method="BOHB-rnn-%d" % id)
    print(bohb.get_incumbent(5))
    return bohb.get_incumbent(5)


def test_hoist(cs, id, scale_mth=6):
    if scale_mth <= 6:
        weight = [0.2]*5
    elif scale_mth == 7:
        weight = [0.0625, 0.125, 0.25, 0.5, 1.0]
    else:
        weight = None
    hoist = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, rho_delta=0.1, enable_rho=True,
                 scale_method=scale_mth, init_weight=weight)
    hoist.runtime_limit = 60000
    method_name = "HOIST-rnn-%d-%d" % (scale_mth, id)
    hoist.method_name = method_name
    hoist.run()
    hoist.plot_statistics(method=method_name)
    print(hoist.get_incumbent(5))
    weights = hoist.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return hoist.get_incumbent(5)


def test_mbhb(cs, id):
    method_name = "MBHB-rnn-%d" % id
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_boes(cs, id):
    method_name = "BOES-rnn-%d" % id
    mbhb = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    mbhb.set_method_name(method_name)
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


def test_baseline_iid(cs, id):
    method_name = "BOSNG-rnn-%d" % id
    mbhb = BaseIID(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


if __name__ == "__main__":
    """
    R = 54 epochs, 480 min each iteration, 170 min
    """
    cs = create_hyperspace()
    test_hoist(cs, 0)
    test_hoist(cs, 1)
