from __future__ import print_function

import sys
sys.setrecursionlimit(10000)

import densenet
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf

from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.callbacks import Callback
# from keras import backend as K
from WM_regularizer import custom_WM_regularizer


def show_ber(model, b):
    weights = model.get_layer('WM_conv').get_weights()
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

    batch_size = 64
    nb_classes = 100
    nb_epoch = 300
    embedded_bits = 128
    img_rows, img_cols = 32, 32
    img_channels = 3

    img_dim = (img_channels, img_rows, img_cols) if tf.keras.backend.image_data_format() == "channels_first" else (img_rows, img_cols, img_channels)
    depth = 100
    nb_dense_block = 3
    growth_rate = 12
    bottleneck = True
    reduction = 0.5
    dropout_rate = 0.2 # 0.0 for data augmentation
    scale = 0.01
    b = np.random.randint(0, 2, size=(1, embedded_bits))
    b_path = 'results/the_watermark_densenet_cifar100_{}bits_uchida.npy'.format(embedded_bits)
    np.save(b_path, b)
    WM = custom_WM_regularizer(b, scale=scale)

    model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                              growth_rate=growth_rate, dropout_rate=dropout_rate,
                              bottleneck=bottleneck, reduction=reduction, weights=None, watermark_regularizer=WM)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-3, epsilon=1e-8) # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    (trainX, trainY), (testX, testY) = cifar100.load_data()

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX /= 255.
    testX /= 255.

    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32)

    generator.fit(trainX, seed=0)

    # Load model
    # model.load_weights("weights/DenseNet-BC-100-12-CIFAR100.h5")
    # print("Model loaded.")

    lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=10, min_lr=0.5e-7)
    early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    model_checkpoint= ModelCheckpoint("weights/DenseNet-BC-100-12-CIFAR100.h5", monitor="val_acc", save_best_only=True,
                                  save_weights_only=True)

    class my_callback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            show_ber()

    ber = my_callback

    callbacks=[lr_reducer, early_stopper, model_checkpoint] # ber

    model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX)/batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, Y_test),
                        verbose=1) # nb_val_samples=testX.shape[0],

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

    weights_path = 'results/densenet_watermarked_cifar100_{}bits_uchida.h5'.format(embedded_bits)
    model.save_weights(weights_path)

    show_ber(model, b)


