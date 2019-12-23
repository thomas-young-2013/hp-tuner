from __future__ import print_function

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
from mfes.utils.ease import ease_target
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.contrib.layers import l2_regularizer, l1_regularizer
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
num_input = 784

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


@ease_target(model_dir="./data/models", name='fcnet')
def train(epochs_num, params, logger=None):
    batch_size = params['batch_size']
    kb_1 = params['kb_1']
    kb_2 = params['kb_2']
    n_layer1 = params['n_layer1']
    n_layer2 = params['n_layer2']
    k_reg = params['k_reg']
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    lr_decay = params['lr_decay']

    need_lc = params['need_lc']
    display_step = 50
    epoch_size = 48000
    iter_num = int(epochs_num) * epoch_size // batch_size
    lc_iter_num = epoch_size // batch_size
    lc_info = []

    # meta file: read and save model file
    read_model_path = params['read_path']
    save_model_path = params['save_path']

    graph = tf.Graph()
    with graph.as_default():
        # tf Graph input
        X = tf.placeholder("float", [None, 28, 28])
        Y = tf.placeholder("float", [None, num_classes])
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        # Create model
        def neural_net(x, kp_1, kp_2, k_reg):
            with tf.variable_scope('FCNet'):
                x = tf.reshape(x, [-1, num_input])
                # Hidden fully connected layer with 256 neurons
                layer_1 = tf.layers.dense(x, n_layer1, kernel_regularizer=l2_regularizer(scale=k_reg))
                layer_1 = tf.nn.dropout(layer_1, kp_1)
                # Hidden fully connected layer with 256 neurons
                layer_2 = tf.layers.dense(layer_1, n_layer2, kernel_regularizer=l2_regularizer(scale=k_reg))
                layer_2 = tf.nn.dropout(layer_2, kp_2)
                # Output fully connected layer with a neuron for each class
                out_layer = tf.layers.dense(layer_2, num_classes)
                return out_layer

        # Construct model
        logits = neural_net(X, keep_prob1, keep_prob2, k_reg)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        # define the RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, decay=lr_decay)
        # add the regularization loss
        loss_reg_op = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'FCNet'))
        train_op = optimizer.minimize(loss_op + loss_reg_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.09)
    config = tf.ConfigProto(gpu_options=gpu_options)
    # config.gpu_options.allow_growth = True
    early_stop = False
    with tf.Session(graph=graph, config=config) as sess:

        def evaluate():
            test_idx = 0
            acc = []
            loss = []
            while test_idx < x_val.shape[0]:
                test_x = x_val[test_idx: test_idx + batch_size]
                test_y = y_val[test_idx: test_idx + batch_size]
                acc_t, loss_t = sess.run([accuracy, loss_op],
                                         feed_dict={X: test_x, Y: test_y, keep_prob1: 1, keep_prob2: 1})
                acc.append(acc_t)
                loss.append(loss_t)
                test_idx += batch_size
            ret_loss = np.mean(loss)
            ret_acc = np.mean(acc)
            return ret_loss, ret_acc

        # Run the initializer
        sess.run(init)
        if os.path.exists(read_model_path + '.meta'):
            saver.restore(sess, read_model_path)
            print('=====================> read model from local file %s' % read_model_path)
            print('=' * 100)

        # Do images classification
        for i in range(1, iter_num + 1):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys, keep_prob1: kb_1, keep_prob2: kb_2})
            # if nan encountered, just break.

            if i % display_step == 0:
                acc, loss = sess.run([accuracy, loss_op], feed_dict={X: batch_xs, Y: batch_ys,
                                                                     keep_prob1: kb_1, keep_prob2: kb_2})
                print(("Step %d" % i) + " batch loss/acc: " + "{:.4f}".format(loss) + "/" + "{:.3f}".format(acc))
                if np.isnan(loss):
                    print('early stop happens!')
                    early_stop = True
                    break

            if need_lc and i % lc_iter_num == 0:
                ret_loss, ret_acc = evaluate()
                lc_info.append(ret_acc)
        ret_loss, ret_acc = evaluate()
        print('Test set => accuracy: %f, loss: %f' % (ret_acc, ret_loss))
        if np.isnan(ret_loss) or ret_loss > 1e5:
            ret_loss = 1e5
            early_stop = True
        save_path = saver.save(sess, save_model_path)
        print("Model saved in file: %s" % save_path)
        return {'loss': 1 - ret_acc, 'early_stop': early_stop, 'lc_info': lc_info}
