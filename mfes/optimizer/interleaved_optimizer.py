import numpy as np

from mfes.optimizer.base_maximizer import BaseOptimizer
from mfes.config_space import get_one_exchange_neighbourhood, sample_configurations
from mfes.config_space import convert_configurations_to_array
from mfes.utils.constants import MAXINT


class InterleavedOptimizer(BaseOptimizer):

    def __init__(self, objective_function, config_space, n_samples=500, rng=None):

        self.n_samples = n_samples
        super(InterleavedOptimizer, self).__init__(objective_function, config_space, rng)

    def maximize(self, batch_size=1):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------
        batch_size: number of maximizer returned.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        incs_configs = list(get_one_exchange_neighbourhood(self.objective_func.eta['config'], seed=self.rng.randint(MAXINT)))
        configs_list = list(incs_configs)
        rand_incs = convert_configurations_to_array(configs_list)

        # Sample random points uniformly over the whole space
        rand_configs = sample_configurations(self.config_space, self.n_samples - rand_incs.shape[0])
        rand = convert_configurations_to_array(rand_configs)

        configs_list.extend(rand_configs)

        X = np.concatenate((rand_incs, rand), axis=0)
        y = self.objective_func(X)
        if batch_size == 1:
            return [configs_list[np.argmax(y)]]

        tmp = configs_list[np.argsort(y)[-batch_size:]]
        return tmp
