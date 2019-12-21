import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=1)
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--iter_c', type=int, default=5)

args = parser.parse_args()

sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
sys.path.append('/home/daim/thomas/run1/hp-tuner')


from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition
from mfes.config_space import ConfigurationSpace
from mfes.evaluate_function.eval_convnet_tf import train
from mfes.facade.bohb import BOHB
from mfes.facade.hb import Hyperband
from mfes.facade.hoist import XFHB
from mfes.facade.batch_bo import SMAC
from mfes.facade.mbhb import MBHB
from mfes.facade.bo_es import SMAC_ES

iter_num = args.iter
maximal_iter = args.R
n_work = args.n
print('training params: R-%d | iter-%d | workers-%d' % (maximal_iter, iter_num, n_work))


def create_hyperspace():
    cs = ConfigurationSpace()

    # training hyperparameters.
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 5e-2, default_value=1e-4, q=3e-5, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", 16, 128, q=16, default_value=32)
    momentum = UniformFloatHyperparameter("momentum", 0., .5, default_value=0., q=.1)
    lr_decay = UniformFloatHyperparameter("lr_decay", .7, .99, default_value=9e-1, q=3e-2)
    dropout_value = UniformFloatHyperparameter("dropout", .1, .7, default_value=.5, q=.1)
    cs.add_hyperparameters([learning_rate, batch_size, momentum, lr_decay, dropout_value])

    # network architecture hyperparameters.
    num_pooling_layer = UniformIntegerHyperparameter("n_pooling_layer", 2, 3, default_value=2)
    num_conv_layer1 = UniformIntegerHyperparameter("n_conv_layer1", 16, 64, default_value=32, q=2)
    num_conv_layer2 = UniformIntegerHyperparameter("n_conv_layer2", 32, 96, default_value=64, q=2)
    num_conv_layer3 = UniformIntegerHyperparameter("n_conv_layer3", 32, 96, default_value=64, q=2)
    num_fully_layer = UniformIntegerHyperparameter("n_fully_unit", 128, 512, default_value=256, q=64)
    cs.add_hyperparameters([num_pooling_layer, num_conv_layer1, num_conv_layer2, num_conv_layer3, num_fully_layer])
    for i in [1, 2, 3]:
        kernel_init_stddev = UniformFloatHyperparameter(
            "kernel_init_stddev%d" % i, 1e-3, 5e-2, default_value=1e-2, q=2e-3)
        kernel_regularizer = UniformFloatHyperparameter(
            "kernel_regularizer%d" % i, 1e-9, 1e-4, default_value=1e-6, q=5e-7, log=True)
        cs.add_hyperparameters([kernel_init_stddev, kernel_regularizer])
        if i == 3:
            k_init_cond = InCondition(child=kernel_init_stddev, parent=num_pooling_layer, values=[3])
            k_reg_cond = InCondition(child=kernel_regularizer, parent=num_pooling_layer, values=[3])
            cs.add_conditions([k_init_cond, k_reg_cond])

    # configuration = cs.get_default_configuration()
    # print(configuration.configuration_space)
    # print(configuration.get_dictionary())
    return cs


def train_convnet(cs):
    print(cs.get_default_configuration().get_dictionary())
    print(train(maximal_iter, cs.get_default_configuration().get_dictionary(), None))


def test_batch_bo(cs, n_id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    bo.run()
    bo.plot_statistics(method="BO-cnn-%d" % n_id)
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_vanilla_bo(cs, n_id):
    bo = SMAC(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    method_name = "Vanilla-BO-cnn-%d" % n_id
    bo.set_method_name(method_name)
    bo.runtime_limit = 50000
    bo.run()
    result = bo.get_incumbent(5)
    print(result)
    return result


def test_hb(cs, n_id):
    hyperband = Hyperband(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    hyperband.run()
    hyperband.plot_statistics(method="HB-cnn-%d" % n_id)
    result = hyperband.get_incumbent(5)
    print(result)
    return result


def test_bohb(cs, n_id):
    bohb = BOHB(cs, train, maximal_iter, num_iter=iter_num, p=0.2, n_workers=n_work)
    bohb.run()
    bohb.plot_statistics(method="BOHB-cnn-%d" % n_id)
    result = bohb.get_incumbent(5)
    print(result)
    return result


def test_hoist(cs, id, scale_mth=6):
    if scale_mth == 6:
        weight = [0.2]*5
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
    hoist.run()
    method_name = "HOIST-cnn-%d-%d" % (scale_mth, id)
    hoist.plot_statistics(method=method_name)
    print(hoist.get_incumbent(5))
    return hoist.get_incumbent(5)


def test_mbhb(cs, n_id):
    mbhb = MBHB(cs, train, maximal_iter, num_iter=iter_num, n_workers=n_work)
    method_name = "MBHB-cnn-%d" % n_id
    mbhb.set_method_name(method_name)
    mbhb.runtime_limit = 50000
    mbhb.run()
    res = mbhb.get_incumbent(5)
    print(res)
    return res


def test_boes(cs, n_id):
    method_name = "BOES-cnn-%d" % n_id
    boes = SMAC_ES(cs, train, maximal_iter, num_iter=iter_num, n_workers=1)
    boes.set_method_name(method_name)
    boes.runtime_limit = 50000
    boes.run()
    print(boes.get_incumbent(5))
    return boes.get_incumbent(5)


if __name__ == "__main__":
    """
    one iteration of hb pipeline takes 120 min
    one iteration of smac takes around 120 min or less
    """
    cs = create_hyperspace()

    # TODO: fix this bug when q = val
    # print(sample_configurations(cs, 10))

    # train_convnet(cs)
    # test_bo(cs, 1)
    # hb_res = test_hb(cs, 1)
    # bohb_res = test_bohb(cs, 1)
    # res = test_bo(cs, 1)
    # res = test_boes(cs, 1)
    # xfhb_res = test_xfhb(cs, 1)
    # test_hoist(cs, 1, scale_mth=7)
    # test_hoist(cs, 2, scale_mth=7)
    # test_hoist(cs, 3, scale_mth=7)
    # test_hoist(cs, 4, scale_mth=7)
    # test_hoist(cs, 5, scale_mth=7)
    # test_boes(cs, 4)
    # test_boes(cs, 5)
    test_mbhb(cs, 7)
    test_mbhb(cs, 8)
