import os
from datetime import datetime
import time
from mfes.evaluate_function.resnet.alexnet import AlexNet
from mfes.evaluate_function.resnet.cifar10_input import *
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
        start_time = time.time()
        epoch_num = int(epoch_num)
        print(epoch_num, params)
        self.train_batch_size = params['train_batch_size']
        self.dropout_rate = params['dropout_rate']
        self.init_lr = params['init_lr']
        self.lr_decay_factor = params['lr_decay_factor']
        self.weight_decay_conv1 = params['weight_decay_conv1']
        self.weight_decay_conv2 = params['weight_decay_conv2']
        self.weight_decay_conv3 = params['weight_decay_conv3']
        self.weight_decay_fc = params['weight_decay_fc']
        self.momentum = params['momentum']
        read_model_path = params['read_path']
        save_model_path = params['save_path']

        # For the first step, we are loading all training images and validation images into the
        # memory
        trainloader = DataLoader(full_dataset, batch_size=self.train_batch_size, num_workers=10, sampler=train_sampler)
        validloader = DataLoader(full_dataset, batch_size=200, num_workers=10, sampler=valid_sampler)

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session

        model = AlexNet(10, self.dropout_rate).to(self.device)
        optimizer = SGD([{'params': model.conv1.parameters(), 'weight_decay': self.weight_decay_conv1},
                         {'params': model.conv2.parameters(), 'weight_decay': self.weight_decay_conv2},
                         {'params': model.conv3.parameters(), 'weight_decay': self.weight_decay_conv3},
                         {'params': model.conv4.parameters(), 'weight_decay': self.weight_decay_conv3},
                         {'params': model.conv5.parameters(), 'weight_decay': self.weight_decay_conv3},
                         {'params': model.classifier.parameters(), 'weight_decay': self.weight_decay_fc}],
                        lr=self.init_lr, momentum=self.momentum)

        scheduler = MultiStepLR(optimizer, milestones=[50, 65],
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

        act_epoch_num = epoch_num
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
        print(time.time() - start_time)
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
