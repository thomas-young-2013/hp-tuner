from __future__ import division, print_function, absolute_import
from mfes.evaluate_function.resnet.cifar100_train import Train
from mfes.utils.ease import ease_target


@ease_target(model_dir="./data/models", name='resnet_cifar100')
def train(epoch_num, params, logger=None):
    try:
        result = Train().train(epoch_num, params, logger)
        print("----------------------")
        print(result)
        print("----------------------")
        return result
    except BaseException as e:
        print(e)
