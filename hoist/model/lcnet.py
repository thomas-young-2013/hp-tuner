import numpy as np
from robo.models.lcnet import LCNet, get_lc_net


class LC_ES(object):
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = LCNet(sampling_method="sghmc",
                           l_rate=np.sqrt(1e-4),
                           mdecay=.05,
                           n_nets=100,
                           burn_in=5000,
                           n_iters=30000,
                           get_net=get_lc_net,
                           precondition=True)
        self.model.train(X, y)

    def predict(self, X, return_individual_predictions=False):
        return self.model.predict(X, return_individual_predictions=return_individual_predictions)
