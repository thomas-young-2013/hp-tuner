from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import l2_regularizer, l1_regularizer
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from hoist.utils.ease import ease_target

epoch_size = 40000
num_classes = 10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape:', x_train.shape) 50000, 32, 32, 3
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# use validation set as evaluation target
x_test, y_test = x_val, y_val
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')


# Create the neural network
def conv_net(x, params, reuse, is_training):
    n_conv_layer = params['n_pooling_layer']
    n_fully_unit = params['n_fully_unit']
    dropout = params['dropout']

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        input = x
        for i in range(1, n_conv_layer + 1):
            conv_unit = params['n_conv_layer%d' % i]
            stddev = params["kernel_init_stddev%d" % i]
            k_reg = params["kernel_regularizer%d" % i]

            # Convolution Layer Group
            conv1_1 = tf.layers.conv2d(input, conv_unit, 3, activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                       kernel_regularizer=l2_regularizer(scale=k_reg))
            conv1_2 = tf.layers.conv2d(conv1_1, conv_unit, 3, activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                       kernel_regularizer=l2_regularizer(scale=k_reg))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1_2, 2, 2)
            input = tf.layers.dropout(conv1, rate=.25, training=is_training)

        # Convolution Layer Group 1
        # conv1_1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
        # conv1_2 = tf.layers.conv2d(conv1_1, 32, 3, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv1 = tf.layers.max_pooling2d(conv1_2, 2, 2)
        # conv1 = tf.layers.dropout(conv1, rate=.25, training=is_training)
        #
        # # Convolution Layer Group 2
        # conv2_1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, padding='same')
        # conv2_2 = tf.layers.conv2d(conv2_1, 64, 3, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
        # conv2 = tf.layers.dropout(conv2, rate=.25, training=is_training)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(input)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, n_fully_unit)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, num_classes)
    return out


# get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


# TODO: consider early-stops
@ease_target(model_dir="./data/models", name='convnet')
def train(epoch_num, params, logger=None):
    import tensorflow as tf

    # training hyperparameters
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    lr_decay = params['lr_decay']
    batch_size = params['batch_size']

    # meta file: read and save model file
    read_model_path = params['read_path']
    save_model_path = params['save_path']

    # running settings
    print('-'*50, batch_size)
    # TODO: hyperspace's bugs assert batch_size != 0
    need_lc = params['need_lc']
    lc_iter_num = epoch_size // batch_size
    num_steps = int(epoch_num)*lc_iter_num
    lc_info = []
    display_step = 500

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.float32, [None, num_classes])

        logits_train = conv_net(X, params, reuse=False, is_training=True)
        logits_test = conv_net(X, params, reuse=True, is_training=False)

        # define softmax loss
        with tf.name_scope('loss'):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
            tf.summary.scalar('loss', loss_op)

        # define the RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, decay=lr_decay)
        # add the regularization loss
        loss_reg_op = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'ConvNet'))
        train_op = optimizer.minimize(loss_op + loss_reg_op)

        # Evaluate the accuracy of the model
        correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start training
    with tf.Session(graph=graph, config=config) as sess:
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("logs/", sess.graph)
        def evaluate():
            test_idx = 0
            acc, loss = [], []
            while test_idx < x_test.shape[0]:
                test_x = x_test[test_idx: test_idx + batch_size]
                test_y = y_test[test_idx: test_idx + batch_size]
                acc_t, loss_t = sess.run([acc_op, loss_op], feed_dict={X: test_x, Y: test_y})
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
            print('='*100)
        early_stop_flag = False
        for step in range(1, num_steps + 1):
            batch_x, batch_y = next_batch(batch_size, x_train, y_train)

            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, acc_op], feed_dict={X: batch_x, Y: batch_y})
                print(("Step %d" % step) + " batch loss/acc: " + "{:.4f}".format(loss) + "/" + "{:.3f}".format(acc))
                if np.isnan(loss):
                    print('early stop happens!')
                    early_stop_flag = True
                    break
                # rs = sess.run(merged, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
                # writer.add_summary(rs, step)
            if need_lc and step % lc_iter_num == 0:
                ret_loss, ret_acc = evaluate()
                lc_info.append(ret_acc)

        ret_loss, ret_acc = evaluate()
        print('Test set => accuracy: %f, loss: %f' % (ret_acc, ret_loss))
        if np.isnan(ret_loss) or ret_loss > 1e5:
            ret_loss = 1e5
        save_path = saver.save(sess, save_model_path)
        print("Model saved in file: %s" % save_path)
        result = {'loss': 1.0-ret_acc, 'early_stop': early_stop_flag, 'lc_info': lc_info}
        return result


if __name__ == "__main__":
    train()
