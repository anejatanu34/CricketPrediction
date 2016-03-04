__author__ = 'anushabala'

from .vgg16 import build_model
import pickle
import lasagne
import numpy as np

model_weights_path = 'vgg16.pkl'


def load_model(path=model_weights_path):
    net = build_model()
    output_layer = net['prob']
    model = pickle.load(open(model_weights_path))
    CLASSES = model['synset words']
    MEAN_IMAGE = np.reshape(model['mean value'], (3,1,1))
    lasagne.layers.set_all_param_values(output_layer, model['param values'])

    return net, output_layer, CLASSES, MEAN_IMAGE