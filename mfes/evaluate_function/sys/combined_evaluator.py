from ConfigSpace import ConfigurationSpace, UnParametrizedHyperparameter, CategoricalHyperparameter
import os
import sys
import time
import pickle as pkl
import numpy as np
from sklearn.metrics.scorer import balanced_accuracy_scorer
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.getcwd())

from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.evaluate_func import partial_validation
from solnml.components.fe_optimizers.ano_bo_optimizer import AnotherBayesianOptimizationOptimizer
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.datasets.utils import load_train_test_data, load_data
from solnml.components.models.classification import _classifiers, _addons

from mfes.utils.ease import ease_target

def get_estimator(config):
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['%s:random_state' % classifier_type] = 1
    hpo_config = dict()
    for key in config_:
        if classifier_type in key:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]
    try:
        estimator = _classifiers[classifier_type](**hpo_config)
    except:
        estimator = _addons.components[classifier_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 1)
    return classifier_type, estimator


def get_hpo_cs(estimator_id, task_type=0):
    if estimator_id in _classifiers:
        clf_class = _classifiers[estimator_id]
    elif estimator_id in _addons.components:
        clf_class = _addons.components[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    cs = clf_class.get_hyperparameter_search_space()
    return cs


def get_fe_cs(estimator_id, node, task_type=0):
    tmp_evaluator = ClassificationEvaluator(None)
    tmp_bo = AnotherBayesianOptimizationOptimizer(task_type, node, tmp_evaluator, estimator_id, 1, 1, 1)
    cs = tmp_bo._get_task_hyperparameter_space('smac')
    return cs


def get_combined_cs(node, task_type=0):
    cs = ConfigurationSpace()
    config_cand = []
    cand_space = {}
    for estimator_id in _classifiers:
        cand_space[estimator_id] = get_hpo_cs(estimator_id, task_type)
        config_cand.append(estimator_id)

    config_option = CategoricalHyperparameter('estimator', config_cand)
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        sub_configuration_space = cand_space[config_item]
        parent_hyperparameter = {'parent': config_option,
                                 'value': config_item}
        cs.add_configuration_space(config_item, sub_configuration_space,
                                   parent_hyperparameter=parent_hyperparameter)
    fe_cs = get_fe_cs(estimator_id, node, task_type)
    for hp in fe_cs.get_hyperparameters():
        cs.add_hyperparameter(hp)
    for cond in fe_cs.get_conditions():
        cs.add_condition(cond)
    for bid in fe_cs.get_forbiddens():
        cs.add_forbidden_clause(bid)
    return cs


def get_fit_params(y, estimator):
    from solnml.components.utils.balancing import get_weights
    _init_params, _fit_params = get_weights(
        y, estimator, None, {}, {})
    return _init_params, _fit_params


tmp_node = load_data('balloon', data_dir='../soln-ml/', task_type=0, datanode_returned=True)
tmp_evaluator = ClassificationEvaluator(None)
tmp_bo = AnotherBayesianOptimizationOptimizer(0, tmp_node, tmp_evaluator, 'adaboost', 1, 1, 1)


@ease_target(model_dir="./data/models", name='sys')
def train(resource_num, params, data_node):
    print(resource_num, params)
    start_time = time.time()
    resource_num = resource_num * 1.0 / 27
    # Prepare data node.
    data_node = data_node['data_node']
    _data_node = tmp_bo._parse(data_node, params)

    X_train, y_train = _data_node.data

    config_dict = params.copy()
    # Prepare training and initial params for classifier.
    init_params, fit_params = {}, {}
    if _data_node.enable_balance == 1:
        init_params, fit_params = get_fit_params(y_train, params['estimator'])
        for key, val in init_params.items():
            config_dict[key] = val

    classifier_id, clf = get_estimator(config_dict)

    try:
        test_size = 0.2
        score = partial_validation(clf, balanced_accuracy_scorer, X_train, y_train, resource_num,
                                   test_size=test_size,
                                   random_state=1,
                                   if_stratify=True,
                                   onehot=None,
                                   fit_params=fit_params)
    except Exception as e:
        print(e)
        score = -np.inf

    print(resource_num, params, -score, time.time() - start_time)
    # Turn it intos a minimization problem.
    return {'loss': -score, 'early_stop': False, 'lc_info': []}
