import numpy as np


class BaseOptimizer(object):

    def __init__(self, objective_function, config_space, rng=None):
        """
        Interface for optimizers that maximizing the
        acquisition function.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        rng: numpy.random.RandomState
            Random number generator
        """
        self.config_space = config_space
        self.objective_func = objective_function
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(10000))
        else:
            self.rng = rng

    def maximize(self):
        pass
