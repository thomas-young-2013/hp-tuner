import os
from datetime import datetime
import time
from mfes.evaluate_function.resnet.resnet import resnet20
from mfes.evaluate_function.resnet.cifar10_input import *
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn


class Train(object):
    def __init__(self):
        # Set up all the placeholders
        self.init_lr = None
        self.train_batch_size = None
        self.weight_decay = None
        self.lr_decay_factor = None
        self.padding_size = None
        self.device = 'cuda'

    def train(self, epoch_num, params, logger=None):
        epoch_num = int(epoch_num)
        print(epoch_num, params)
        self.train_batch_size = params['train_batch_size']
        self.init_lr = params['init_lr']
        self.lr_decay_factor = params['lr_decay_factor']
        self.weight_decay = params['weight_decay']
        self.momentum = params['momentum']
        self.nesterov = True if params['nesterov'] == 'True' else False
        read_model_path = params['read_path']
        save_model_path = params['save_path']

        # For the first step, we are loading all training images and validation images into the
        # memory
        trainloader = DataLoader(full_dataset, batch_size=self.train_batch_size, num_workers=10, sampler=train_sampler)
        validloader = DataLoader(full_dataset, batch_size=200, num_workers=10, sampler=valid_sampler)

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session

        model = resnet20(10).to(self.device)
        optimizer = SGD(params=model.parameters(), lr=self.init_lr, momentum=self.momentum,
                        weight_decay=self.weight_decay, nesterov=self.nesterov)

        scheduler = MultiStepLR(optimizer, milestones=[68, 102],
                                gamma=self.lr_decay_factor)
        loss_func = nn.CrossEntropyLoss()

        # If you want to load from a checkpoint
        if os.path.exists(read_model_path):
            checkpoint = torch.load(read_model_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch_num']
            print('=====================> read model from local file %s' % read_model_path)
            print('=' * 100)
        else:
            start_epoch = 0

        print('Start training...')
        print('----------------------------')

        act_epoch_num = 5 * epoch_num
        lc_info = []

        for epoch_id in range(start_epoch, start_epoch + act_epoch_num):
            model.train()
            # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_avg_loss = 0
            epoch_avg_acc = 0
            val_avg_loss = 0
            val_avg_acc = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, data in enumerate(trainloader):
                batch_x, batch_y = data[0], data[1]
                num_train_samples += len(batch_x)
                logits = model(batch_x.float().to(self.device))
                loss = loss_func(logits, batch_y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
                prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

            epoch_avg_loss /= num_train_samples
            epoch_avg_acc /= num_train_samples

            print('Epoch %d: Train loss %.4f, train acc %.4f' % (epoch_id, epoch_avg_loss, epoch_avg_acc))

            if validloader is not None:
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(validloader):
                        batch_x, batch_y = data[0], data[1]
                        logits = model(batch_x.float().to(self.device))
                        val_loss = loss_func(logits, batch_y.to(self.device))
                        num_val_samples += len(batch_x)
                        val_avg_loss += val_loss.to('cpu').detach() * len(batch_x)

                        prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                        val_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

                    val_avg_loss /= num_val_samples
                    val_avg_acc /= num_val_samples
                    print('Epoch %d: Val loss %.4f, val acc %.4f' % (epoch_id, val_avg_loss, val_avg_acc))

            scheduler.step()

        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'epoch_num': start_epoch + act_epoch_num}
        torch.save(state, save_model_path)
        print("Model saved in file: %s" % save_model_path)
        result = {'loss': 1 - val_avg_acc, 'early_stop': False, 'lc_info': lc_info}
        return result

    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance
        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // TEST_BATCH_SIZE
        remain_images = num_test_images % TEST_BATCH_SIZE
        print('%i test batches in total...' % num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[TEST_BATCH_SIZE,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, NUM_RESIDUAL_BLOCKS, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session(config=self.config)

        saver.restore(sess, TEST_CKPT_PATH)
        print('Model restored from ', TEST_CKPT_PATH)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print('%i batches finished!' % step)
            offset = step * TEST_BATCH_SIZE
            test_image_batch = test_image_array[offset:offset + TEST_BATCH_SIZE, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, NUM_RESIDUAL_BLOCKS, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array

    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset + vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset + vali_batch_size]
        return vali_data_batch, vali_label_batch

    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset + train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=self.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset + train_batch_size]

        return batch_data, batch_label

    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(TRAIN_EMA_DECAY, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=self.momentum)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op

    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op

    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        VALID_SIZE = 10000
        num_batches = VALID_SIZE // VALID_BATCH_SIZE
        order = np.random.choice(VALID_SIZE, num_batches * VALID_BATCH_SIZE)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * VALID_BATCH_SIZE
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                         self.vali_image_placeholder: vali_data_subset[offset:offset + VALID_BATCH_SIZE,
                                                      ...],
                         self.vali_label_placeholder: vali_labels_subset[offset:offset + VALID_BATCH_SIZE],
                         self.lr_placeholder: self.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)
