import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--iter', type=int, default=5)
parser.add_argument('--iter_c', type=int, default=5)

args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/hp-tuner')

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition
from hoist.config_space import ConfigurationSpace
from hoist.evaluate_function.eval_reinforce_tf import train
from hoist.facade.bohb import BOHB
from hoist.facade.hb import Hyperband
from hoist.facade.hoist import XFHB
from hoist.facade.batch_bo import SMAC

iter_num = args.iter
maximal_iter = args.R
n_work = args.n
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def create_hyperspace():
    cs = ConfigurationSpace()

    # training hyperparameters.
    policy_learning_rate = UniformFloatHyperparameter("policy_learning_rate", 1e-3, 0.5, default_value=1e-2, q=3e-3)
    discount_factor = UniformFloatHyperparameter("discount_factor", 0.9, 1., q=0.01, default_value=1.)
    value_learning_rate = UniformFloatHyperparameter("value_learning_rate", 1e-3, 0.5, default_value=.1, q=3e-3)
    cs.add_hyperparameters([policy_learning_rate, value_learning_rate, discount_factor])
    return cs


def train_rl(cs):
    print(cs.get_default_configuration().get_dictionary())
    print(train(maximal_iter, cs.get_default_configuration().get_dictionary(), None))


def test_bo(cs, n_id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.run()
    bo.plot_statistics(method="BO-rl-%d" % n_id)
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_hb(cs, n_id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.run()
    hyperband.plot_statistics(method="HB-rl-%d" % n_id)
    result = hyperband.get_incumbent(5)
    print(result)
    return result


def test_bohb(cs, n_id):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.run()
    bohb.plot_statistics(method="BOHB-rl-%d" % n_id)
    result = bohb.get_incumbent(5)
    print(result)
    return result


def test_hoist(cs, n_id, update_w=True):
    weight = [1.0, 0.5, 0.25, 0.125, 0.125]
    hoist = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted',
                 update_enable=update_w, update_delta=1, random_mode=False, init_weight=weight)
    hoist.run()
    method_name = "HOIST-rl-%d" % n_id
    if not update_w:
        method_name = "HOIST-rl-no_update-%d" % n_id
    hoist.plot_statistics(method=method_name)
    result = hoist.get_incumbent(5)
    print(result)
    return result


def test_methods_mean_variance(cs):
    iterations = args.iter_c
    for i in range(1, 1+iterations):
        weight = [1.0, 0.5, 0.25, 0.125, 0.125]
        bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
        bo.run()
        bo.plot_statistics(method="BO-rl-%d" % i)

        hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
        hyperband.set_restart()
        hyperband.run()
        hyperband.plot_statistics(method="HB-rl-%d" % i)

        bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
        bohb.set_restart()
        bohb.run()
        bohb.plot_statistics(method="BOHB-rl-%d" % i)

        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted')
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="HOIST-rl-no_update-%d" % i)

        xfhb = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work, info_type='Weighted',
                    update_enable=True, update_delta=1, random_mode=False, init_weight=weight)
        xfhb.set_restart()
        xfhb.run()
        xfhb.plot_statistics(method="HOIST-rl-%d" % i)


if __name__ == "__main__":
    cs = create_hyperspace()
    # train_rl(cs, 1)
    # test_bohb(cs, 1)
    test_hoist(cs, 1)
    # test_methods_mean_variance(cs)
