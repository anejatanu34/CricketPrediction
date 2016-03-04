__author__ = 'anushabala'

from .vgg16 import build_model
import os
import pickle
import lasagne
import urllib
import numpy as np
import io
import skimage.transform
import matplotlib.pyplot as plt
from lasagne.utils import floatX

model_weights_path = 'vgg16.pkl'
net = build_model()
output_layer = net['prob']
model = pickle.load(open(model_weights_path))
CLASSES = model['synset words']
lasagne.layers.set_all_param_values(output_layer, model['param values'])