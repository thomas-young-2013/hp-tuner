from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter


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
    elif benchmark_id == 'xgb':
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
    elif benchmark_id == 'resnet':
        cs = ConfigurationSpace()
        padding_size = CategoricalHyperparameter('padding_size', [1, 2, 3], default_value=2)
        # batch_size = CategoricalHyperparameter('train_batch_size', [256])
        batch_size = UniformIntegerHyperparameter("train_batch_size", 32, 256, default_value=64, q=8)
        init_lr = UniformFloatHyperparameter('init_lr', lower=1e-3, upper=0.3, default_value=0.1, log=True)
        lr_decay_factor = UniformFloatHyperparameter('lr_decay_factor', lower=0.01, upper=0.2, default_value=0.1,
                                                     log=True)
        weight_decay = UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, default_value=0.0002,
                                                  log=True)
        momentum = UniformFloatHyperparameter("momentum", 0.5, .99, default_value=0.9)
        cs.add_hyperparameters([padding_size, batch_size, init_lr, lr_decay_factor, weight_decay, momentum])
    else:
        raise ValueError('Invalid benchmark id: %s!' % benchmark_id)
    return cs
