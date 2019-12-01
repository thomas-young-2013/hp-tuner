import numpy as np
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.model_selection import train_test_split

# Import ConfigSpace and different types of parameters
from hoist.config_space import ConfigurationSpace
from hoist.facade.random_search import RandomSearch


def try_params(params):
    raw_data = load_breast_cancer()
    train_x, x_test, train_y, y_test = train_test_split(raw_data.data,
                                                        raw_data.target,
                                                        test_size=0.2,
                                                        random_state=0)

    clf = RF(n_estimators=25, verbose=0, n_jobs=-1, **params)
    clf.fit(train_x, train_y)

    try:
        p = clf.predict_proba(x_test)[:, 1]  # sklearn convention
    except IndexError:
        p = clf.predict_proba(x_test)

    ll = log_loss(y_test, p)
    auc = AUC(y_test, p)
    acc = accuracy(y_test, np.round(p))

    print("# testing  | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format(ll, auc, acc))

    return {'loss': ll, 'log_loss': ll, 'auc': auc}

# print(cs.get_default_configuration())
# print(cs.sample_configuration(2))


cs = ConfigurationSpace()

max_depth = UniformIntegerHyperparameter("max_depth", 2, 10, default_value=5)
cs.add_hyperparameter(max_depth)

min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 10, default_value=5)
cs.add_hyperparameter(min_samples_split)


hyperband = RandomSearch(cs, try_params, num_trails=100)

print(hyperband.run())
