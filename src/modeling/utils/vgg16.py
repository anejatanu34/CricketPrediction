# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers.recurrent import LSTMLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne import nonlinearities
from lasagne.layers import DimshuffleLayer
from lasagne.layers import SliceLayer

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    try:
        from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
    except ImportError:
        try:
            from lasagne.layers import Conv2DLayer as ConvLayer
        except ImportError:
            # this should never happen
            raise ImportError("Could not find required Lasagne components")
from lasagne.nonlinearities import softmax
from lasagne.init import HeNormal, Constant

MAX_GRAD = 100


def build_model(output_neurons=1000):
    net = {'input': InputLayer((None, 3, 224, 224))}
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=output_neurons, W=HeNormal(), b=Constant(0.0), nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


def build_late_fusion_model(output_neurons=4):
    # conv layers identical to VGGNet
    net = {'input': InputLayer((None, 3, 224, 224))}
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    # Reshape conv layer output of shape (N,F,H,W) to (N/2, F*2, H, W)
    # i.e. combine every two frames into one along the "channels" dimension to combine information
    conv_out_shape = list(net['pool5'].output_shape)
    conv_out_shape[1] *= 2
    conv_out_shape[0] = -1
    net['rshp5_4'] = ReshapeLayer(net['pool5'], conv_out_shape)
    # Pass reshaped volume to first FC layer
    net['fc6'] = DenseLayer(net['rshp5_4'],  num_units=4096, W=HeNormal(gain='relu'), b=Constant(0.0))
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096, W=HeNormal(gain='relu'), b=Constant(0.0))
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=output_neurons, W=HeNormal(), b=Constant(0.0), nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


def build_lstm_classification_model(output_neurons=4, num_frames=5, hidden_units=100):
    # conv layers identical to VGGNet
    net = {'input': InputLayer((None, 3, 224, 224))}
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    net['fc6'] = DenseLayer(net['pool5'],  num_units=4096, W=HeNormal(gain='relu'), b=Constant(0.0))
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096, W=HeNormal(gain='relu'), b=Constant(0.0))
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    fc_out = net['fc7_dropout'].output_shape
    net['reshape'] = ReshapeLayer(net['fc7_dropout'], (-1, num_frames, fc_out[1]))
    net['lstm'] = LSTMLayer(net['reshape'], num_units=hidden_units, grad_clipping=MAX_GRAD,
                            nonlinearity=nonlinearities.tanh)
    net['slice_lstm'] = SliceLayer(net['lstm'], -1, 1)
    net['fc8'] = DenseLayer(net['slice_lstm'], num_units=output_neurons,
                            W=HeNormal(), b=Constant(0.0), nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net
