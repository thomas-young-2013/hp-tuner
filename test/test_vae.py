import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--iter_c', type=int, default=5)

args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/running3/hp-tuner')

# Import ConfigSpace and different types of parameters
from mfes.config_space import ConfigurationSpace, sample_configurations
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from mfes.evaluate_function.eval_vae_tf import train
from mfes.facade.bohb import BOHB
from mfes.facade.hb import Hyperband
from mfes.facade.hoist import XFHB
from mfes.facade.batch_bo import SMAC
from mfes.facade.mbhb import MBHB


iter_num = args.iter
maximal_iter = args.R
n_work = args.n
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def train_vae(cs):
    conf = cs.get_default_configuration().get_dictionary()
    conf['need_lc'] = False
    print(train(81, conf, None))


def test_hb(cs, id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.run()
    hyperband.plot_statistics(method="HB-vae-%d" % id)
    print(hyperband.get_incumbent(5))
    return hyperband.get_incumbent(5)


def test_bo(cs, id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.run()
    bo.plot_statistics(method="BO-vae-%d" % id)
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_bohb(cs, id):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.run()
    bohb.plot_statistics(method="BOHB-vae-%d" % id)
    print(bohb.get_incumbent(5))
    return bohb.get_incumbent(5)


def test_hoist(cs, id):
    # weight = [0.533, 0.267, 0.133, 0.0667, 0.0333]
    weight = [0.2, 0.2, 0.2, 0.2, 0.2]
    hoist = XFHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work,
                 update_enable=True, rho_delta=0.1, enable_rho=False, init_rho=0.5,
                 scale_method=6, init_weight=weight)
    hoist.run()
    method_name = "HOIST-vae-%d" % id
    hoist.plot_statistics(method=method_name)
    print(hoist.get_incumbent(5))
    return hoist.get_incumbent(5)


def test_mbhb(cs, id):
    method_name = "MBHB-vae-%d" % id
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    mbhb.set_method_name(method_name)
    mbhb.run()
    mbhb.plot_statistics(method=method_name)
    print(mbhb.get_incumbent(5))
    return mbhb.get_incumbent(5)


if __name__ == "__main__":
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-4, 5e-2, default_value=1e-3, q=2e-4, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", 32, 256, default_value=64, q=16)
    n_layer1 = UniformIntegerHyperparameter("hidden_units", 256, 768, default_value=512, q=16)
    n_layer2 = UniformIntegerHyperparameter("latent_units", 2, 20, default_value=5, q=1)

    cs.add_hyperparameters([learning_rate, n_layer1, n_layer2, batch_size])

    # train_vae(cs)
    # test_hb(cs, 1)
    test_hoist(cs, 1)
