import numpy as np
from keras.datasets import cifar100
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 100

TRAIN_RANDOM_LABEL = False  # Want to use random label for train data?
VALI_RANDOM_LABEL = False  # Want to use random label for validation?

EPOCH_SIZE = 40000

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def load_dataset(type='train'):
    if type == 'train':
        return x_train, y_train.reshape(-1)
    elif type == 'val':
        return x_val, y_val.reshape(-1)
    elif type == 'test':
        return x_test, y_test.reshape(-1)
    else:
        raise ValueError('Invalid dataset type!')


def read_in_all_images(shuffle=True):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data, label = load_dataset(type='train')
    num_data = data.shape[0]
    if shuffle is True:
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image


def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset + IMG_HEIGHT,
                                y_offset:y_offset + IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def prepare_train_data(padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''
    data, label = read_in_all_images()
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

    return data, label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = load_dataset(type='val')
    # print(validation_array.shape, validation_labels.shape)
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels
