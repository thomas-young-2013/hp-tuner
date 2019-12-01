import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--n', type=int, default=6)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--runtime_limit', type=int, default=21600)

args = parser.parse_args()

sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
sys.path.append('/home/daim/thomas/hp-tuner')
sys.path.append('/home/liyang/codes/hp-tuner')
sys.path.append('/home/liyang/thomas/hp-tuner')

# Import ConfigSpace and different types of parameters
from hoist.config_space import ConfigurationSpace, sample_configurations
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from hoist.evaluate_function.eval_xgb import train
from hoist.facade.bohb import BOHB
from hoist.facade.hb import Hyperband
from hoist.facade.hoist import XFHB
from hoist.facade.hoist_opt import HOIST
from hoist.facade.batch_bo import SMAC
from hoist.facade.baseline_iid import BaseIID


benchmark = 'xgb'
iter_num = args.iter
maximal_iter = args.R
n_work = args.n
runtime_limit = args.runtime_limit
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def train_xgb(cs):
    conf = cs.get_default_configuration().get_dictionary()
    conf['need_lc'] = False
    print(train(1, conf, None))


def test_batch_bo(cs, id):
    model = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    model.method_name = "BO-xgb-%d" % id
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    result = model.get_incumbent(5)
    print(result)
    return result


def test_vanilla_bo(cs, id):
    model = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    model.method_name = "Vanilla-BO-xgb-%d" % id
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    result = model.get_incumbent(5)
    print(result)
    return result


def test_hb(cs, id):
    model = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    model.method_name = "HB-xgb-%d" % id
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    result = model.get_incumbent(5)
    print(result)
    return result


def test_bohb(cs, id):
    model = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    model.method_name = "BOHB-xgb-%d" % id
    model.runtime_limit = runtime_limit
    model.restart_needed = True
    model.run()
    result = model.get_incumbent(5)
    print(result)
    return result


def test_hoist(cs, id, scale_mth=6):
    if scale_mth <= 6:
        weight = [0.2]*5
    elif scale_mth == 7:
        weight = [0.0625, 0.125, 0.25, 0.5, 1.0]
    else:
        weight = None
    model = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, rho_delta=0.1, enable_rho=True,
                 scale_method=scale_mth, init_weight=weight)

    method_name = "HOIST-xgb-%d-%d" % (scale_mth, id)
    model.method_name = method_name
    model.runtime_limit = 27000
    model.restart_needed = True
    model.run()
    print(model.get_incumbent(5))
    weights = model.get_weights()
    np.save('data/weights_%s.npy' % method_name, np.asarray(weights))
    return model.get_incumbent(5)


def create_configspace():
    cs = ConfigurationSpace()
    # n_estimators = UniformFloatHyperparameter("n_estimators", 100, 600, default_value=200, q=10)
    eta = UniformFloatHyperparameter("eta", 0.01, 0.9, default_value=0.3, q=0.01)
    min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, default_value=1, q=0.1)
    max_depth = UniformIntegerHyperparameter("max_depth", 1, 12, default_value=6)
    subsample = UniformFloatHyperparameter("subsample", 0.1, 1, default_value=1, q=0.1)
    gamma = UniformFloatHyperparameter("gamma", 0, 10, default_value=0, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, default_value=1., q=0.1)
    alpha = UniformFloatHyperparameter("alpha", 0, 10, default_value=0., q=0.1)
    _lambda = UniformFloatHyperparameter("lambda", 1, 10, default_value=1, q=0.1)

    cs.add_hyperparameters([eta, min_child_weight, max_depth, subsample, gamma,
                            colsample_bytree, alpha, _lambda])
    return cs


if __name__ == "__main__":
    cs = create_configspace()

    # Test the objective function.
    # train_xgb(cs)
    # test_vanilla_bo(cs, 1)
    # test_batch_bo(cs, 1)
    # test_hb(cs, 1)
    # test_bohb(cs, 1)
    # test_old_hoist(cs, 1)
    test_hoist(cs, 0)
    test_hoist(cs, 1)
