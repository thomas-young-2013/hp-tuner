import sys

if int(sys.argv[1]) == 0:
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/hp-tuner')

from mfes.utils.ease import ease_target


@ease_target(model_dir="./data/models", name='convnet')
def train(iter_num, params, logger=None):
    if logger is not None:
        print(logger)

    print(params)
    # result = {'loss': 123, 'ref_id': 123}
    result = {'loss': 123}
    return result


if __name__ == "__main__":
    params = {'dropout': 0.123455, 'learning_rate': 1.012334, 'reference': 'ref_xxxxx'}
    res = train(123, params, logger=1234)
    print(res)
