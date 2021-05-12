from tensorflow.python.keras.regularizers import Regularizer
import tensorflow as tf
import numpy as np


class custom_WM_regularizer(Regularizer):

    def __init__(self, b, scale):
        self.scale = tf.keras.backend.cast_to_floatx(scale)
        self.b = b

    def __call__(self, x):
        w_shape = x.shape
        pro_matrix_rows = np.prod(w_shape[0:3])
        pro_matrix_cols = self.b.shape[1]
        self.pro_matrix_value = np.random.randn(pro_matrix_rows, pro_matrix_cols)
        pro_matrix_path = 'results/projection_matrix.npy'
        np.save(pro_matrix_path, self.pro_matrix_value)
        regularization = 0
        w_mean = tf.keras.backend.mean(x, axis=3)
        w_mean = tf.keras.backend.reshape(w_mean, (1, tf.keras.backend.count_params(w_mean)))
        projection_matrix = tf.keras.backend.variable(value=self.pro_matrix_value)
        # regularization += self.scale * tf.keras.backend.sum(
            # tf.keras.losses.binary_crossentropy(1 / (1 + tf.keras.backend.exp( (-10) * tf.keras.backend.dot(w_mean, projection_matrix))), tf.convert_to_tensor(self.b, tf.float32)))
        regularization += self.scale * tf.keras.backend.sum(tf.keras.losses.binary_crossentropy(
            tf.keras.backend.exp(10 * tf.keras.backend.sin(10 * tf.keras.backend.dot(w_mean, projection_matrix))) / (1 + tf.keras.backend.exp(
                10 * tf.keras.backend.sin(10 * tf.keras.backend.dot(w_mean, projection_matrix)))), tf.convert_to_tensor(self.b, tf.float32)))
        return regularization

