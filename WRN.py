# This code is imported from the following project: https://github.com/titu1994/Wide-Residual-Networks

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization

from tensorflow.python.keras import backend as K



def initial_conv(input):
    x = Convolution2D(16, (3, 3), padding='same')(input)

    channel_axis = 1 if K.image_data_format() == "th" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def conv1_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_data_format() == "th" else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == "th":
        if init.shape[1] != 16 * k:
            init = Convolution2D(16 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init.shape[-1] != 16 * k:
            init = Convolution2D(16 * k, (1, 1), activation='linear', padding='same')(init)

    x = Convolution2D(16 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_data_format() == "th" else -1

    # Check if input number of filters is same as 32 * k, else create convolution2d for this input
    if K.image_data_format() == "th":
        if init.shape[1] != 32 * k:
            init = Convolution2D(32 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init.shape[-1] != 32 * k:
            init = Convolution2D(32 * k, (1, 1), activation='linear', padding='same')(init)

    x = Convolution2D(32 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_data_format() == "th" else -1

    # Check if input number of filters is same as 64 * k, else create convolution2d for this input
    if K.image_data_format() == "th":
        if init.shape[1] != 64 * k:
            init = Convolution2D(64 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init.shape[-1] != 64 * k:
            init = Convolution2D(64 * k, (1, 1), activation='linear', padding='same')(init)

    x = Convolution2D(64 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1, wmark_regularizer=None,
                                 target_blk_num=1):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """

    def get_regularizer(blk_num, idx):
        if wmark_regularizer != None and target_blk_num == blk_num and idx == 0:
            # print('target regularizer({}, {})'.format(blk_num, idx))
            return wmark_regularizer
        else:
            return None

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    for i in range(N):
        x = conv1_block(x, k, dropout, get_regularizer(1, i))
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = conv2_block(x, k, dropout, get_regularizer(2, i))
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = conv3_block(x, k, dropout, get_regularizer(3, i))
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


if __name__ == "__main__":

    init = (32, 32, 3)