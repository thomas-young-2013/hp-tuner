from __future__ import division, print_function, absolute_import

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from mfes.utils.ease import ease_target

# Training Parameters
# learning_rate = 0.001
# batch_size = 48
# lstm_units = 64
# keep_prob_value = .75

epoch_size = 20000
num_classes = 2
max_seq_length = 250
data_dir = 'data/lstm_data/'

# load word embedding data.
wordsList = np.load(data_dir + 'wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load(data_dir + 'wordVectors.npy')
print('Loaded the word vectors!')
print(len(wordsList))
print(wordVectors.shape)
ids = np.load(data_dir+'idsMatrix.npy')
print(ids.shape)


def determine_max_length(type='train'):
    cwd = os.getcwd()
    os.chdir(data_dir+type)
    from os import listdir
    from os.path import isfile, join
    positive_files = ['pos/' + f for f in listdir('pos/') if isfile(join('pos/', f))]
    negative_files = ['neg/' + f for f in listdir('neg/') if isfile(join('neg/', f))]
    num_words = []
    for pf in positive_files:
        with open(pf, 'r') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print('Positive files finished')

    for nf in negative_files:
        with open(nf, 'r') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print('Negative files finished')

    os.chdir(cwd)

    num_files = len(num_words)
    print('The total number of files is', num_files)
    print('The total number of words in the files is', sum(num_words))
    print('The average number of words in the files is', sum(num_words) / len(num_words))
    return positive_files, negative_files


def create_ids_matrix(positive_files, negative_files, type='train'):
    cwd = os.getcwd()
    os.chdir(data_dir+type)

    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    def clean_sentences(string):
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())

    ids = np.zeros((25000, max_seq_length), dtype='int32')
    file_counter = 0
    for pf in positive_files:
        print('file counter', file_counter)
        with open(pf, "r") as f:
            index_counter = 0
            line = f.readline()
            cleaned_line = clean_sentences(line)
            split = cleaned_line.split()
            for word in split:
                try:
                   ids[file_counter][index_counter] = wordsList.index(word)
                except ValueError:
                   ids[file_counter][index_counter] = 399999  # vector for unknown words
                index_counter = index_counter + 1
                if index_counter >= max_seq_length:
                   break
            file_counter = file_counter + 1

    for nf in negative_files:
        print('file counter', file_counter)
        with open(nf, "r") as f:
            index_counter = 0
            line=f.readline()
            cleaned_line = clean_sentences(line)
            split = cleaned_line.split()
            for word in split:
                try:
                    ids[file_counter][index_counter] = wordsList.index(word)
                except ValueError:
                    ids[file_counter][index_counter] = 399999  # vector for unknown words
                index_counter = index_counter + 1
                if index_counter >= max_seq_length:
                    break
            file_counter = file_counter + 1
    os.chdir(cwd)
    np.save(data_dir + ('idsMatrix_%s' % type), ids)


# ids_train_x = np.load(data_dir+'idsMatrix_train.npy')
ids_train_x = np.load(data_dir+'idsMatrix.npy')
# ids_test_x = np.load(data_dir+'idsMatrix_test.npy')
ids_train_y = np.zeros((ids_train_x.shape[0], 2))
ids_train_y[:12500, 0] = 1
ids_train_y[12500:, 1] = 1
# ids_test_y = ids_train_y

train_x, val_x, train_y, val_y = train_test_split(ids_train_x, ids_train_y, test_size=0.2, random_state=42)
print(train_x.shape, train_y.shape)
print(val_x.shape)


# get a batch of data.
def next_batch(num, data, labels):
    import random
    idx = random.sample(range(len(data)), num)
    data_batch = [data[i] for i in idx]
    labels_batch = [labels[i] for i in idx]
    return np.asarray(data_batch), np.asarray(labels_batch)


@ease_target(model_dir="./data/models", name='lstm')
def train(epoch_num, params, logger=None):
    import tensorflow as tf

    # training hyperparameters
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    lstm_units = params['lstm_units']
    keep_prob_value = params['keep_prob']

    # meta file: read and save model file
    read_model_path = params['read_path']
    save_model_path = params['save_path']

    # running settings
    print('-' * 50, batch_size)
    need_lc = params['need_lc']
    display_step = 200
    lc_iter_num = epoch_size // batch_size
    num_steps = int(epoch_num) * 2 * lc_iter_num
    lc_info = []

    graph = tf.Graph()
    with graph.as_default():
        keep_prob = tf.placeholder(tf.float32)
        labels = tf.placeholder(tf.float32, [None, num_classes])
        input_data = tf.placeholder(tf.int32, [None, max_seq_length])
        data = tf.nn.embedding_lookup(wordVectors, input_data)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)
        value, state = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        # value = tf.transpose(value, [1, 0, 2])
        # last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(state[1], weight) + bias)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start training
    with tf.Session(graph=graph, config=config) as sess:
        def evaluate():
            test_idx = 0
            acc_list, loss_list = [], []
            while test_idx < val_x.shape[0]:
                margin_idx = test_idx + batch_size
                next_batch_data, next_batch_labels = val_x[test_idx: margin_idx], val_y[test_idx: margin_idx]
                acc_t, loss_t = sess.run([accuracy, loss],
                                         {input_data: next_batch_data, labels: next_batch_labels, keep_prob: 1.0})
                acc_list.append(acc_t)
                loss_list.append(loss_t)
                test_idx = margin_idx
            acc_value, loss_value = np.mean(acc_list), np.mean(loss_list)
            return loss_value, acc_value

        # Run the initializer
        sess.run(init)
        if os.path.exists(read_model_path + '.meta'):
            start_time = time.time()
            saver.restore(sess, read_model_path)
            print('=====================> read model from local file %s' % read_model_path)
            print('it takes %.4f' % (time.time() - start_time))
            print('=' * 100)

        early_stop_flag = False
        for i in range(1, 1 + num_steps):
            next_batch_data, next_batch_labels = next_batch(batch_size, train_x, train_y)
            sess.run(optimizer, {input_data: next_batch_data, labels: next_batch_labels, keep_prob: keep_prob_value})
            if i % display_step == 0:
                acc_t, loss_t = sess.run([accuracy, loss],
                    {input_data: next_batch_data, labels: next_batch_labels, keep_prob: keep_prob_value})
                print('%d step loss/acc: %.4f/%.4f' % (i, loss_t, acc_t))
                if np.isnan(loss_t):
                    print('early stop happens!')
                    early_stop_flag = True
                    break
            if need_lc and i % lc_iter_num == 0:
                ret_loss, ret_acc = evaluate()
                lc_info.append(ret_acc)

        loss_value, acc_value = evaluate()
        print('Test set => accuracy: %f, loss: %f' % (acc_value, loss_value))
        if np.isnan(loss_value) or loss_value > 1e5:
            loss_value = 1e5
        save_path = saver.save(sess, save_model_path)
        print("Model saved in file: %s" % save_path)
        result = {'loss': 1. - acc_value, 'early_stop': early_stop_flag, 'lc_info': lc_info}
        return result
