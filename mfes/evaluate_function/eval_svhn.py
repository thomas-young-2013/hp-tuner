from __future__ import division, print_function, absolute_import
from mfes.evaluate_function.resnet.shvn_train import Train
# from mfes.evaluate_function.resnet.svhn_test import Train as Test
from mfes.utils.ease import ease_target


@ease_target(model_dir="./data/models", name='svhn')
def train(epoch_num, params, logger=None):
    try:
        result = Train().train(epoch_num, params, logger)
        print("----------------------")
        print(result)
        print("----------------------")
        return result
    except BaseException as e:
        print(e)


@ease_target(model_dir="./data/models", name='svhn')
def eval(epoch_num, params, logger=None):
    try:
        result = Test().train(epoch_num, params, logger)
        print("----------------------")
        print(result)
        print("----------------------")
        return result
    except BaseException as e:
        print(e)
