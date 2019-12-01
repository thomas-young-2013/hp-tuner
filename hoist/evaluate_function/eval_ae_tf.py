from __future__ import division, print_function, absolute_import

from scipy.stats import norm
import tensorflow as tf
import numpy as np
import os
from hoist.utils.ease import ease_target
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (-1, 784))
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


@ease_target(model_dir="./data/models", name='autoencoder')
def train(epochs_num, params, logger=None):
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']

    display_step = 500
    epoch_size = 48000
    iter_num = int(epochs_num)*epoch_size // batch_size
    lc_iter_num = epoch_size // batch_size
    lc_info = []

    # # meta file: read and save model file
    read_model_path = params['read_path']
    save_model_path = params['save_path']
    need_lc = params['need_lc']

    # Network Parameters
    num_hidden_1 = params['num_h1']  # 1st layer num features
    num_hidden_2 = params['num_h2']  # 2nd layer num features (the latent dim)
    num_input = 784  # MNIST data input (img shape: 28*28)

    # A custom initialization (see Xavier Glorot init)
    def glorot_init(shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    graph = tf.Graph()
    with graph.as_default():
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, num_input])
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([num_input])),
        }

        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                           biases['encoder_b1']))
            # Encoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                           biases['encoder_b2']))
            return layer_2

        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                           biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                           biases['decoder_b2']))
            return layer_2

        # Construct model
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    early_stop = False
    with tf.Session(graph=graph, config=config) as sess:
        # Run the initializer
        sess.run(init)
        if os.path.exists(read_model_path + '.meta'):
            saver.restore(sess, read_model_path)
            print('=====================> read model from local file %s' % read_model_path)
            print('='*100)

        def evaluate():
            test_idx = 0
            loss_list = []
            while test_idx < x_val.shape[0]:
                test_x = x_val[test_idx: test_idx + batch_size]
                loss_t = sess.run(loss, feed_dict={X: test_x})
                loss_list.append(loss_t)
                test_idx += batch_size
            return np.mean(loss_list)

        # Training
        for i in range(1, iter_num + 1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, _ = next_batch(batch_size, x_train, y_train)
            # batch_x, _ = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            if np.isnan(l):
                print('early stop happens!')
                early_stop = True
                break
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

            if need_lc and i % lc_iter_num == 0:
                ret_loss = evaluate()
                lc_info.append(1 - ret_loss)

        ret_loss = evaluate()
        print('Test set => loss: %f' % ret_loss)
        if np.isnan(ret_loss) or ret_loss > 1e5:
            ret_loss = 1e5
            early_stop = True
        save_path = saver.save(sess, save_model_path)
        print("Model saved in file: %s" % save_path)
        return {'loss': ret_loss, 'early_stop': early_stop, 'lc_info': lc_info}
