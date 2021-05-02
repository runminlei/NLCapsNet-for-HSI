import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.regularizers import l2

K.set_image_data_format('channels_last')

from keras.layers import Activation, Reshape, Lambda, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K


class Mish(layers.Layer):
    """
    Mish Activation Function.

    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
        tanh = (1 - e^{-2x}) / (1 + e^{-2x})

    # Shape:
        -Input: Arbitrary. Use the keyword argument 'input_shape' (tuple of integers, does not include the samples axis)
                when using this layer as the first layer in a model.
        -Output: Same shape as the input.

    Examples:
        X_input = Input(input_shape)
        X = Mish()(X_input)

        X = conv2d(...)(input)
        X = Mish()(X)
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).

    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.

    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x


class CapsnetBuilder(object):
    @staticmethod
    def build(input_shape, n_class, routings):
        """
        A Capsule Network
        :param input_shape: data shape, 3d, [width, height, channels]
        :param n_class: number of classes
        :param routings: number of routing iterations
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                'eval_model' can also be used for training.
        """
        x = layers.Input(shape=input_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', kernel_initializer='he_normal',
                              # kernel_regularizer=l2(0.0001),
                              name='conv1')(x)
        # conv1 = layers.BatchNormalization(momentum=0.9, name='bn1')(conv1)
        conv1 = layers.Activation('relu')(conv1)
        # conv1 = Mish()(conv1)
        conv1 = non_local_block(conv1, compression=1)

        # Layer 2: Just a conventional Conv2D layer
        conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', kernel_initializer='he_normal',
                              # kernel_regularizer=l2(0.0001),
                              name='conv2')(conv1)
        # conv2 = layers.BatchNormalization(momentum=0.9, name='bn2')(conv2)
        conv2 = layers.Activation('relu')(conv2)
        # conv2 = Mish()(conv2)
        # conv2 = non_local_block(conv2, compression=1)

        # Layer 3: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=32, kernel_size=3, strides=2, padding='valid')

        # Layer 4: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                 name='digitcaps')(primarycaps)

        out_caps = Length(name='capsnet')(digitcaps)

        # Decoder network.
        y = layers.Input(shape=(n_class,))
        masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
        decoder.add(layers.Dense(1024, activation='relu'))
        decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

        # Models for training and evaluation (prediction)
        train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = models.Model(x, out_caps)

        return train_model, eval_model

    @staticmethod
    def build_capsnet(input_shape, n_class, routings):
        return CapsnetBuilder.build(input_shape, n_class, routings)


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def main():
    model, eval_model = CapsnetBuilder.build_capsnet(input_shape=(27, 27, 3), n_class=9, routings=3)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.0005],
                  metrics={'capsnet': 'accuracy'})
    model.summary()


if __name__ == '__main__':
    main()