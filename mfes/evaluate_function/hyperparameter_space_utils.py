from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition


def get_benchmark_configspace(benchmark_id):
    if benchmark_id == 'fcnet':
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
    elif benchmark_id in ['covtype', 'higgs']:
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
    elif benchmark_id in ['cifar', 'svhn']:
        cs = ConfigurationSpace()
        # padding_size = CategoricalHyperparameter('padding_size', [1, 2, 3], default_value=2)
        # batch_size = CategoricalHyperparameter('train_batch_size', [256])
        batch_size = UniformIntegerHyperparameter("train_batch_size", 32, 256, default_value=64, q=8)
        init_lr = UniformFloatHyperparameter('init_lr', lower=1e-3, upper=0.3, default_value=0.1, log=True)
        lr_decay_factor = UniformFloatHyperparameter('lr_decay_factor', lower=0.01, upper=0.2, default_value=0.1,
                                                     log=True)
        weight_decay = UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, default_value=0.0002,
                                                  log=True)
        momentum = UniformFloatHyperparameter("momentum", 0.5, .99, default_value=0.9)
        nesterov = CategoricalHyperparameter('nesterov', ['True', 'False'], default_value='True')
        cs.add_hyperparameters([nesterov, batch_size, init_lr, lr_decay_factor, weight_decay, momentum])
    elif benchmark_id == 'convnet':
        cs = ConfigurationSpace()

        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 5e-2, default_value=1e-4, q=3e-5, log=True)
        batch_size = UniformIntegerHyperparameter("batch_size", 16, 128, q=16, default_value=32)
        momentum = UniformFloatHyperparameter("momentum", 0., .5, default_value=0., q=.1)
        lr_decay = UniformFloatHyperparameter("lr_decay", .7, .99, default_value=9e-1, q=3e-2)
        dropout_value = UniformFloatHyperparameter("dropout", .1, .7, default_value=.5, q=.1)
        cs.add_hyperparameters([learning_rate, batch_size, momentum, lr_decay, dropout_value])

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

        return cs
    else:
        raise ValueError('Invalid benchmark id: %s!' % benchmark_id)
    return cs
