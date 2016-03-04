__author__ = 'anushabala'

from .vgg16 import build_model
import pickle
import lasagne
import numpy as np

DEFAULT_MODEL_PATH = 'vgg16.pkl'


class VGGModel(object):
    net = None
    labels = None
    mean_bgr = None
    model_weights = None

    def __init__(self, path=DEFAULT_MODEL_PATH, classes_key='synset words', mean_key='mean value', weights_key='param values'):
        self.net = build_model()
        model = pickle.load(open(path))
        self.labels = model[classes_key]
        self.mean_bgr = model[mean_key]
        self.model_weights = model[weights_key]

    def output_layer(self):
        return self.net['prob']

    def model(self):
        return self.net

    def _set_model_params(self):
        lasagne.layers.set_all_param_values(self.output_layer(), self.model_weights)


def load_model():
    net = build_model()
    output_layer = net['prob']
    model = pickle.load()
    CLASSES = model['synset words']
    MEAN_IMAGE = np.reshape(model['mean value'], (3,1,1))
    lasagne.layers.set_all_param_values(output_layer, model['param values'])

    return net, output_layer, CLASSES, MEAN_IMAGE