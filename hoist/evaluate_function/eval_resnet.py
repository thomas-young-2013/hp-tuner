from __future__ import division, print_function, absolute_import
from hoist.evaluate_function.resnet.cifar10_train import Train
from hoist.utils.ease import ease_target


@ease_target(model_dir="./data/models", name='resnet')
def train(epoch_num, params, logger=None):
    try:
        result = Train().train(epoch_num, params, logger)
        print("----------------------")
        print(result)
        print("----------------------")
        return result
    except BaseException as e:
        print(e)
