from __future__ import division, print_function, absolute_import
import os
import sys
from functools import partial

sys.path.append(os.getcwd())
from solnml.datasets.utils import load_train_test_data

from mfes.evaluate_function.sys.combined_evaluator import train as _train

train_node, test_node = load_train_test_data('electricity', data_dir='../soln-ml/', task_type=0)
train = partial(_train, data_node=train_node)
