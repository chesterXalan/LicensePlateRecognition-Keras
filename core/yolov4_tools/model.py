"""YOLO_v4 Model Defined in Keras."""
from functools import wraps, reduce

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model

class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

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

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/"""
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['kernel_initializer'] = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)
    route = Concatenate()([postconv, shortconv])
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def yolov4(inputs, num_anchors, num_classes):
    """Create YOLO_V4 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))

    #19x19 head
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(darknet.output)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(y19)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(y19)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(y19)
    y19 = Concatenate()([maxpool1, maxpool2, maxpool3, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)

    y19_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(y19)

    #38x38 head
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(darknet.layers[204].output)
    y38 = Concatenate()([y38, y19_upsample])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(y38)

    #76x76 head
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(darknet.layers[131].output)
    y76 = Concatenate()([y76, y38_upsample])
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)

    #76x76 output
    y76_output = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y76_output)

    #38x38 output
    y76_downsample = ZeroPadding2D(((1,0),(1,0)))(y76)
    y76_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))(y76_downsample)
    y38 = Concatenate()([y76_downsample, y38])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_output = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y38_output)

    #19x19 output
    y38_downsample = ZeroPadding2D(((1,0),(1,0)))(y38)
    y38_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(y38_downsample)
    y19 = Concatenate()([y38_downsample, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)

    y19_output = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y19_output)

    yolo4_model = Model(inputs, [y19_output, y38_output, y76_output])

    return yolo4_model
