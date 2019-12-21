import numpy as np
from mfes.config_space import convert_configurations_to_array

def init_random_sample(config_space, n_points):
    """
    Samples N configs uniformly.

    Parameters
    ----------
    config_space: ConfigurationSpace
        Configuration space
    n_points: int
        The number of initial data points
    rng: np.random.RandomState
            Random number generator
    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """

    configurations = config_space.sample_configuration(n_points)
    X = []
    for config in configurations:
        X.append(config.get_array())

    return np.array(X)
