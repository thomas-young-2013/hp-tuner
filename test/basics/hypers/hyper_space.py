import os
import sys
sys.path.append(os.getcwd())

# Import ConfigSpace and different types of parameters
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition
from mfes.config_space import ConfigurationSpace
from mfes.config_space import convert_configurations_to_array, get_random_neighborhood
from mfes.config_space import get_one_exchange_neighbourhood, get_random_neighbor
from mfes.utils.util_funcs import get_types
from mfes.config_space.util import get_hp_neighbors, get_configuration_id
from mfes.config_space.util import sample_configurations
cs = ConfigurationSpace()
criterion = CategoricalHyperparameter("criterion", ["gini", "entropy", "hehe"], default_value="gini")
cs.add_hyperparameter(criterion)

bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=False)
cs.add_hyperparameter(bootstrap)

max_depth = UniformIntegerHyperparameter("max_depth", 2, 10, default_value=5)
cs.add_hyperparameter(max_depth)

# print(cs.get_default_configuration().get_array())
# print(cs.sample_configuration(2))
# print(type(cs.sample_configuration(2)))
conf = cs.get_default_configuration()

# print(get_random_neighborhood(conf, 4, 123))
print(conf.get_array())
print(conf.keys())
print(conf.get_dictionary())
# print(conf.configuration_space)
# print(conf.configuration_space.get_hyperparameters())

# c = conf.get_dictionary()
# array = conf.get_array()
# print(array)
# for hp in conf.configuration_space.get_hyperparameters():
#     data = hp._inverse_transform(c[hp.name])
#     neis = hp.get_neighbors(data, np.random.RandomState(122), 4, True)
#     print(neis)


# print(conf)
# space = conf.configuration_space
# print(space)
# print(get_random_neighbor(space.get_default_configuration(), 12))

# print(conf.get_dictionary())
# print('the maximal depth is: %d' % conf.get('max_depth'))

# conf_vector = convert_configurations_to_array([conf])[0]
# print(conf_vector)
# conf  = list(get_one_exchange_neighbourhood(conf, 1))[0]
# print(conf != conf)
# print(conf)

# for item in get_one_exchange_neighbourhood(conf, int(1e8)):
#     print(item)

def test_neigh():
    space = conf.configuration_space
    neighbours = list(get_one_exchange_neighbourhood(conf, int(1e8)))
    neighbours += neighbours
    print(neighbours)
    print(neighbours[0] == neighbours[1])
    print(neighbours[3] == neighbours[2])
    print(conf in neighbours)
    print(neighbours.count(neighbours[0]))

# neighbours = list(neighbours)
# neig = convert_configurations_to_array(neighbours)
# print(neig.shape[0])
# print(get_types(cs))

# -----uniform float hyperparameter-------
# lower = 0.1
# upper = 10
# f1 = UniformFloatHyperparameter("param", lower, upper, q=0.5, log=False)
# print(f1.sample(np.random))
# r1 = f1._inverse_transform(np.array([2, 8]))
# print(r1)
# print(f1.get_neighbors(r1[1], np.random.RandomState(122), 4, True))
# # print(f1.get_neighbors())
# print(f1.is_legal(3.0))
# print(f1.is_legal(3))


# -----uniform integer hyperparameter-------
def test_hp():
    cs = ConfigurationSpace()
    lower = 32
    upper = 64
    f1 = UniformIntegerHyperparameter("param", lower, upper, default_value=64, log=False)
    criterion = CategoricalHyperparameter("criterion", ["gini", "entropy", "hehe"], default_value="hehe")
    cs.add_hyperparameter(criterion)
    cs.add_hyperparameter(f1)
    configuration = cs.get_default_configuration()
    print(get_configuration_id(configuration))
    # print(get_random_neighborhood(configuration, 10, 123))
    dict_data = configuration.get_dictionary()
    configuration_space = configuration.configuration_space
    for item in configuration_space.get_hyperparameters():
        print(item)
        print(get_hp_neighbors(item, dict_data, 7, transform=False))
        # print(item.get_neighbors())

# print(f1.sample(np.random))
# r1 = f1._inverse_transform(np.array([38, 55]))
# print(r1)
# print(f1.get_neighbors(r1[1], np.random.RandomState(122), 4, True))
# test_hp()

# -----categorical hyperparameter------
# f1 = CategoricalHyperparameter("criterion", ["gini", "entropy", "gini-entropy"], default_value="gini")
# print(f1.default_value)
# r1 = f1._inverse_transform('gini-entropy')
# print(r1)
# print(f1.get_neighbors(0, np.random.RandomState(122), 4, False))

# TEST: get_hp_neighbors
# cs = ConfigurationSpace()
# learning_rate = UniformFloatHyperparameter("learning_rate", 5e-3, 5e-2, default_value=3e-2, q=5e-3)
# # learning_rate = UniformFloatHyperparameter("learning_rate", 5e-3, 5e-2, default_value=3e-2, q=5e-3)
# cs.add_hyperparameter(learning_rate)
# configuration = cs.get_default_configuration()
# data_dict = configuration.get_dictionary()
# # # print(get_random_neighbor(configuration, 3))
# # # print(get_random_neighborhood(configuration, 3, 123))
#
# configuration_space = configuration.configuration_space
# for item in configuration_space.get_hyperparameters():
#     print(item)
#     print(get_hp_neighbors(item, data_dict, 7))
#     # print(item.get_neighbors())

# conf_dict_data = configuration.get_dictionary()
# h_num = len(conf_dict_data)
# array_data = configuration.get_array()
# neighbor_dict = dict()
# for key, value in conf_dict_data.items():
#     neighbor_dict[key] = [array_data[configuration_space._hyperparameter_idx[key]]]


def test_uniform_log_parameter():
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 5e-1, default_value=1e-3, q=1e-4, log=True)
    cs.add_hyperparameter(learning_rate)
    configuration = cs.get_default_configuration()
    print(configuration)
    print(get_random_neighbor(configuration, 3))
    print(get_random_neighborhood(configuration, 3, 123))
    print(list(get_one_exchange_neighbourhood(configuration, 123)))


def test_create_hyperspace():
    cs = ConfigurationSpace()

    # training hyperparameters.
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 5e-1, default_value=1e-3, q=3e-5, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", 8, 128, q=16, default_value=32)
    lr_momentum = UniformFloatHyperparameter("lr_momentum", 1e-5, 5e-1, default_value=1e-3, q=3e-5, log=True)
    lr_decay = UniformFloatHyperparameter("lr_decay", 1e-5, 5e-1, default_value=1e-3, q=3e-5, log=True)
    dropout_value = UniformFloatHyperparameter("dropout", 0, 1., default_value=.5, q=.1)
    cs.add_hyperparameters([learning_rate, batch_size, lr_momentum, lr_decay, dropout_value])

    # network architecture hyperparameters.
    num_pooling_layer = UniformIntegerHyperparameter("n_pooling_layer", 2, 3, default_value=2)
    num_conv_layer1 = UniformIntegerHyperparameter("n_conv_layer1", 16, 64, default_value=32, q=2)
    num_conv_layer2 = UniformIntegerHyperparameter("n_conv_layer2", 16, 64, default_value=32, q=2)
    num_conv_layer3 = UniformIntegerHyperparameter("n_conv_layer3", 16, 64, default_value=32, q=2)
    cs.add_hyperparameters([num_pooling_layer, num_conv_layer1, num_conv_layer2, num_conv_layer3])
    for i in [1, 2, 3]:
        kernel_init_stddev = UniformFloatHyperparameter(
            "kernel_init_stddev%d" % i, 1e-3, 5e-2, default_value=1e-2, q=2e-3)
        kernel_regularizer = UniformFloatHyperparameter(
            "kernel_regularizer%d" % i, 1e-7, 1e-3, default_value=1e-5, q=5e-7)
        activity_regularizer = UniformFloatHyperparameter(
            "activity_regularizer%d" % i, 1e-7, 1e-3, default_value=1e-5, q=5e-7)
        cs.add_hyperparameters([kernel_init_stddev, kernel_regularizer, activity_regularizer])
        if i == 3:
            k_init_cond = InCondition(child=kernel_init_stddev, parent=num_pooling_layer, values=[3])
            k_reg_cond = InCondition(child=kernel_regularizer, parent=num_pooling_layer, values=[3])
            ac_reg_cond = InCondition(child=activity_regularizer, parent=num_pooling_layer, values=[3])
            cs.add_conditions([k_init_cond, k_reg_cond, ac_reg_cond])

    configuration = cs.get_default_configuration()
    # print(configuration.get_array())
    # print(convert_configurations_to_array([configuration]))
    configs = sample_configurations(cs, 5)
    mappings = dict()
    for i, conf in enumerate(configs):
        mappings[conf] = i
    print(mappings[configs[4]])
    # print(configuration.configuration_space)
    # print(configuration.get_dictionary())
    return cs


def test_sample_configurations():
    cs = test_create_hyperspace()
    config = cs.get_default_configuration()
    print(config)
    print(get_random_neighborhood(config, 10, 12))


def test_log_parameter():
    cs = ConfigurationSpace()
    # learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 5e-1, default_value=1e-4, q=3e-5, log=True)
    kernel_regularizer = UniformFloatHyperparameter(
        "kernel_regularizer", 1e-9, 1e-4, default_value=1e-6, q=5e-7, log=True)
    cs.add_hyperparameter(kernel_regularizer)
    # print(cs.sample_configuration(10))
    print(sample_configurations(cs, 10))
    # config = cs.get_default_configuration()
    # print((config, 10, random.randint))
    # print(config)


if __name__ == "__main__":
    # test_uniform_log_parameter()
    test_create_hyperspace()
    # test_sample_configurations()
    # test_log_parameter()
