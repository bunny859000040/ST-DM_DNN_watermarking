import numpy as np
# import pandas as pd
import sys
import json
import os
import sklearn.metrics as metrics
import WRN as wrn
import tensorflow.python.keras
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import callbacks as callbacks
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import LearningRateScheduler
from WM_regularizer import custom_WM_regularizer

lr_schedule = [60, 120, 160]


def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.01
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.002  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.0004
    return 0.00008

def show_ber(model, b):
    weights = model.get_layer('convolution2d_10').get_weights()
    weight = (np.array(weights[0])).mean(axis=3)
    pred_bparam = tf.keras.backend.dot(tf.keras.backend.variable(value=weight.reshape(1, weight.size)), tf.keras.backend.variable(value=WM.pro_matrix_value))
    pred_bparam = tf.keras.backend.exp(10 * tf.keras.backend.sin(10 * pred_bparam)) / (1 + tf.keras.backend.exp(10 * tf.keras.backend.sin(10 * pred_bparam)))
    # pred_bparam = 1 / (1 + tf.keras.backend.exp((-10) * pred_bparam))
    pred_bparam[pred_bparam >= 0.5] = 1
    pred_bparam[pred_bparam < 0.5] = 0
    diff = np.abs(pred_bparam - b[1, :])
    print("error bits num = ", np.sum(diff))
    BER = np.sum(diff) / b.shape[1]
    print("BER = ", BER)

if __name__ == '__main__':

    settings_json_fname = 'config/train.json'
    MODEL_CHKPOINT_FNAME = 'result/WRN_checkpoint.h5'
    train_settings = json.load(open(settings_json_fname))

    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    trainY = np_utils.to_categorical(trainY)
    testY = np_utils.to_categorical(testY)

    generator = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)
    generator.fit(trainX, seed=0, augment=True)

    batch_size = train_settings['batch_size']
    nb_epoch = train_settings['epoch']
    scale = train_settings['scale']
    embed_dim = train_settings['embed_dim']
    N = train_settings['N']
    k = train_settings['k']
    target_blk_id = train_settings['target_blk_id']
    base_modelw_fname = train_settings['base_modelw_fname']

    b = np.random.randint(0, 2, size=(1, embed_dim))
    b_path = 'result/WRN_CIFAR10_the_watermark_{}bits.npy'.format(embed_dim)

    WM = custom_WM_regularizer(b, scale)

    init_shape = (3, 32, 32) if K.image_data_format() == 'th' else (32, 32, 3)

    model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=N, k=k, dropout=0.00,
                                             wmark_regularizer=WM, target_blk_num=target_blk_id)

    model.summary()
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])

    if len(base_modelw_fname) > 0:
        model.load_weights(base_modelw_fname)

    model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX),
                        nb_epoch=nb_epoch,
                        callbacks=[
                            callbacks.ModelCheckpoint(MODEL_CHKPOINT_FNAME, monitor="val_acc", save_best_only=True),
                            LearningRateScheduler(schedule=schedule)
                        ],
                        validation_data=(testX, testY),
                        nb_val_samples=testX.shape[0], )

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yPred = np_utils.to_categorical(yPred)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

    show_ber(model, b)
