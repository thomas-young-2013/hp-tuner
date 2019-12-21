from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
from mfes.utils.ease import ease_target
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_classes = 10

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def deepnn(inputs, keep_prob, n_layer1=32, n_layer2=64, n_fc=512):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inputs, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, n_layer1])
        b_conv1 = bias_variable([n_layer1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, n_layer1, n_layer2])
        b_conv2 = bias_variable([n_layer2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * n_layer2, n_fc])
        b_fc1 = bias_variable([n_fc])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_layer2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([n_fc, 10])
        b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    res = tf.nn.softmax(y_conv)
    return res


# get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


def train(epochs_num, params, logger=None):
    batch_size = 64
    learning_rate = 1e-3
    keep_pb = 0.5
    display_step = 100
    n_layer1 = 32
    n_layer2 = 64
    n_fc = 512
    if 'batch_size' in params:
        batch_size = int(params['batch_size'])
    if 'lr' in params:
        learning_rate = params['lr']
    if 'keep_prob' in params:
        keep_pb = params['keep_prob']
    if 'fc_unit' in params:
        n_fc = int(params['fc_unit'])

    epoch_size = 9600

    basic_iter = epoch_size // batch_size
    iter_num = 27 * basic_iter
    check_iter_ids = [basic_iter*i for i in [1, 3, 9, 27]]

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder("float", [None, 28, 28])
        y_ = tf.placeholder("float", [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        y = deepnn(X, keep_prob, n_layer1, n_layer2, n_fc=n_fc)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    early_stop = False
    with tf.Session(graph=graph, config=config) as sess:

        def evaluate():
            test_idx = 0
            acc = []
            loss = []
            while test_idx < x_val.shape[0]:
                test_x = x_val[test_idx: test_idx + batch_size]
                test_y = y_val[test_idx: test_idx + batch_size]
                acc_t, loss_t = sess.run([accuracy, cross_entropy], feed_dict={X: test_x, y_: test_y, keep_prob: 1.})
                acc.append(acc_t)
                loss.append(loss_t)
                test_idx += batch_size
            ret_loss = np.mean(loss)
            ret_acc = np.mean(acc)
            return ret_loss, ret_acc

        # Run the initializer
        sess.run(init)
        val_list = []
        # Do images classification
        for i in range(1, iter_num+1):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys, keep_prob: keep_pb})
            # if nan encountered, just break.

            if i % display_step == 0:
                acc, loss = sess.run([accuracy, cross_entropy], feed_dict={X: batch_xs, y_: batch_ys, keep_prob: keep_pb})
                print(("Step %d" % i) + " batch loss/acc: " + "{:.4f}".format(loss) + "/" + "{:.3f}".format(acc))
                if np.isnan(loss):
                    print('early stop happens!')
                    early_stop = True
                    break
            if i in check_iter_ids:
                ret_loss, ret_acc = evaluate()
                val_list.append(1-ret_acc)

        ret_loss, ret_acc = evaluate()
        if early_stop:
            val_list += [1-ret_acc]*(len(check_iter_ids)-len(val_list))
        print('Test set => error: %s' % str(val_list))
        if np.isnan(ret_loss) or ret_loss > 1e5:
            early_stop = True
        return {'loss': val_list, 'early_stop': early_stop}
