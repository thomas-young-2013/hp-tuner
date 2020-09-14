import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
VALID_BATCH_SIZE = 250
TEST_BATCH_SIZE = 125
NUM_RESIDUAL_BLOCKS = 5
IS_FULL_VALID = True
TRAIN_EMA_DECAY = 0.95
TEST_CKPT_PATH = 'model_110.ckpt-79999'
VERSION = 'test_110'

train_dir = 'logs_' + VERSION + '/'
