from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
from mfes.utils.ease import ease_target
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_val = x_test


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
    epoch_size = 60000
    iter_num = 4*int(epochs_num)*epoch_size // batch_size
    lc_iter_num = epoch_size // batch_size
    lc_info = []

    # # meta file: read and save model file
    read_model_path = params['read_path']
    save_model_path = params['save_path']
    need_lc = params['need_lc']

    # Network Parameters
    image_dim = 784
    hidden_dim = params['hidden_units']
    latent_dim = params['latent_units']

    graph = tf.Graph()
    with graph.as_default():
        # A custom initialization (see Xavier Glorot init)
        def glorot_init(shape):
            return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

        # Variables
        weights = {
            'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
            'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
            'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
            'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
            'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
        }
        biases = {
            'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
            'z_mean': tf.Variable(glorot_init([latent_dim])),
            'z_std': tf.Variable(glorot_init([latent_dim])),
            'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
            'decoder_out': tf.Variable(glorot_init([image_dim]))
        }

        # Building the encoder
        input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
        encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
        encoder = tf.nn.tanh(encoder)
        z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
        z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

        # Sampler: Normal (gaussian) random distribution
        eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                               name='epsilon')
        z = z_mean + tf.exp(z_std / 2) * eps

        # Building the decoder (with scope to re-use these layers later)
        decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
        decoder = tf.nn.tanh(decoder)
        decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
        decoder = tf.nn.sigmoid(decoder)

        # Define VAE Loss
        def vae_loss(x_reconstructed, x_true):
            # Reconstruction loss
            encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                                 + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
            encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
            # KL Divergence loss
            kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
            return tf.reduce_mean(encode_decode_loss + kl_div_loss)/1000

        loss_op = vae_loss(decoder, input_image)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
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
                loss_t = sess.run(loss_op, feed_dict={input_image: test_x})
                loss_list.append(loss_t)
                test_idx += batch_size
            return np.mean(loss_list)

        # Training
        for i in range(1, iter_num + 1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, _ = next_batch(batch_size, x_train, y_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([train_op, loss_op], feed_dict={input_image: batch_x})
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
        if np.isnan(ret_loss) or ret_loss > 1.0:
            ret_loss = 1.0
            early_stop = True
        save_path = saver.save(sess, save_model_path)
        print("Model saved in file: %s" % save_path)
        return {'loss': ret_loss, 'early_stop': early_stop, 'lc_info': lc_info}
