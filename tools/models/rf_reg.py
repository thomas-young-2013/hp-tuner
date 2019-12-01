import numpy as np
from hoist.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from pyrfr import regression

from hoist.model.rf_with_instances import RandomForestWithInstances
from hoist.utils.util_funcs import get_types

if __name__ == "__main__":
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-4, 5e-3, default_value=3e-4)
    cs.add_hyperparameter(learning_rate)

    n_layer1 = UniformIntegerHyperparameter("n_layer1", 5, 50, default_value=32)
    cs.add_hyperparameter(n_layer1)

    n_layer2 = UniformIntegerHyperparameter("n_layer2", 30, 80, default_value=64)
    cs.add_hyperparameter(n_layer2)

    batch_size = UniformIntegerHyperparameter("batch_size", 10, 500, default_value=200)
    cs.add_hyperparameter(batch_size)

    types, bounds = get_types(cs)
    reg = regression.binary_rss_forest()
    rf_opts = regression.forest_opts()
    rf_opts.num_trees = 10
    rf_opts.do_bootstrapping = True

    model = RandomForestWithInstances(types=types, bounds=bounds)
    x = np.array([[0.78105907, 0.33860037, 0.72826097, 0.02941158],
                  [0.81160897, 0.63147998, 0.72826097, 0.04901943],
                  [0.27800406, 0.36616871, 0.16304333, 0.24509794],
                  [0.41242362, 0.37351241, 0.11956505, 0.4607843],
                  [0.70162934, 0.15819312, 0.51086957, 0.10784298],
                  [0.53869654, 0.86662495, 0.27173903, 0.22549009],
                  [0.53665988, 0.68576624, 0.81521753, 0.06862728],
                  [0.72199594, 0.18900731, 0.75000011, 0.36274504]], dtype=np.float64)
    y = np.array([0.544481, 2.34456, 0.654629, 0.576376, 0.603501, 0.506214, 0.416664, 0.483639])
    print(x.dtype)
    rf_opts.num_data_points_per_tree = x.shape[0]
    reg.options = rf_opts
    model.train(x, y)
    print(model.predict(x))
    # data = regression.default_data_container(x.shape[1])
    # for row_X, row_y in zip(x, y):
    #     data.add_data_point(row_X, row_y)
    # reg.fit(data, rng=regression.default_random_engine(123))
    #
    # print(reg.predict_mean_var(np.array([0.27800406, 0.36616871, 0.16304333, 0.24509794])))
