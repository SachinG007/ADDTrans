from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.initializers import _compute_fans
from keras.optimizers import SGD
from keras import backend as K
import keras

WEIGHT_DECAY = 0.5 * 0.0005


class SGDTorch(SGD):
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m + g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p - lr * (self.momentum * v + g)
            else:
                new_p = p - lr * v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


def _get_channels_axis():
    return -1 if K.image_data_format() == 'channels_last' else 1


def _conv_kernel_initializer(shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = np.sqrt(2. / fan_in)
    return K.random_normal(shape, 0., stddev, dtype)


def _dense_kernel_initializer(shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = 1. / np.sqrt(fan_in)
    return K.random_uniform(shape, -stddev, stddev, dtype)
    
def batch_norm():
    return BatchNormalization(axis=_get_channels_axis(), momentum=0.1, epsilon=1e-4,
                              beta_regularizer=None, gamma_regularizer=None)


def conv2d(output_channels, kernel_size, strides=1):
    return Convolution2D(output_channels, kernel_size, strides=strides, padding='same', use_bias=False,
                         kernel_initializer=_conv_kernel_initializer, kernel_regularizer=l2(WEIGHT_DECAY))


def dense(output_units):
    return Dense(output_units, kernel_initializer=_dense_kernel_initializer, kernel_regularizer=l2(WEIGHT_DECAY),
                 bias_regularizer=l2(WEIGHT_DECAY))


def lenet(input_shape, num_classes, depth, widen_factor=1, dropout_rate=0.0,
                                 final_activation='softmax'):
    n_channels = [32, 64, 128]

    inp = Input(shape=input_shape)
    conv1 = Convolution2D(n_channels[0], 5, padding="same")(inp)  # one conv at the beginning (spatial size: 32x32)
    conv1 = MaxPooling2D(pool_size=(2, 2))(Activation('relu')(batch_norm()(conv1)))

    conv2 = Convolution2D(n_channels[1], 5, padding="same")(conv1)  # one conv at the beginning (spatial size: 32x32)
    conv2 = MaxPooling2D(pool_size=(2, 2))(Activation('relu')(batch_norm()(conv2)))

    conv3 = Convolution2D(n_channels[2], 5, padding="same")(conv2)  # one conv at the beginning (spatial size: 32x32)
    conv3 = MaxPooling2D(pool_size=(2, 2))(Activation('relu')(batch_norm()(conv3)))
    
    out = Flatten()(conv3)
    out = Activation(final_activation)(dense(num_classes)(out))

    return Model(inp, out)
