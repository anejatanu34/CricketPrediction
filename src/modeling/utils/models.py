__author__ = 'anushabala'

from .vgg16 import build_model, get_ordered_layers
import pickle
import lasagne
import numpy as np
import theano.tensor as T


class Outcome(object):
    RUN = 0
    NO_RUN = 1
    BOUNDARY = 2
    OUT = 3
    ILLEGAL = 4

    @classmethod
    def class_labels(cls):
        return [cls.NO_RUN, cls.RUN, cls.BOUNDARY, cls.OUT]

    @classmethod
    def get_label_from_commentary(cls, outcome):
        if outcome == '1' or outcome == '2':
            return cls.RUN
        if outcome == 'no run' or outcome == 'no_run':
            return cls.NO_RUN
        if outcome == 'out':
            return cls.OUT
        if outcome == 'boundary':
            return cls.BOUNDARY

        return None

    @classmethod
    def name(cls, label):
        if label == cls.RUN:
            return 'run'
        if label == cls.NO_RUN:
            return 'no run'
        if label == cls.BOUNDARY:
            return 'boundary'
        if label == cls.OUT:
            return 'out'


class Model(object):
    net = None
    labels = None
    mean_bgr = None
    model_weights = None
    output_key = 'prob'

    def output_layer(self):
        return self.net[self.output_key]

    def model(self):
        return self.net

    def layer(self, key):
        return self.net[key]

    def get_output(self, input_image, mode='train', layer_name=output_key):
        if mode == 'test':
            return lasagne.layers.get_output(self.layer(layer_name), input_image, deterministic=True)
        else:
            return lasagne.layers.get_output(self.layer(layer_name), input_image, deterministic=False)


class VGG16Model(Model):
    def __init__(self, path, classes_key='synset words', mean_key='mean value', weights_key='param values'):
        self.net = build_model()
        model = pickle.load(open(path))
        self.labels = model[classes_key]
        self.mean_bgr = np.reshape(model[mean_key], (3,1,1))
        self.model_weights = model[weights_key]

        self._set_model_params()

    def _set_model_params(self):
        lasagne.layers.set_all_param_values(self.output_layer(), self.model_weights)


class AverageFrameModel(Model):
    tuning_layers = []

    def __init__(self, path, mean_key='mean value',
                 weights_key='param values', output_neurons=4, tuning_layers=None, class_labels=Outcome.class_labels()):
        self.net = build_model(output_neurons)
        model = pickle.load(open(path))
        self.labels = class_labels
        self.mean_bgr = np.reshape(model[mean_key], (3,1,1))
        self.model_weights = model[weights_key]
        self.tuning_layers = tuning_layers

        self._set_model_params()

    def _set_model_params(self):
        """
        Set params according to VGG16 pretrained weights for all layers except output layer
        """
        last_layer = self.net['fc8'].input_layer
        lasagne.layers.set_all_param_values(last_layer, self.model_weights[:-2])

    def clip_loss(self, frames, y, mode='train'):
        """
        Get loss for a single clip by computing the average of the loss for each frame
        :param frames: All the frames in a single clip (4D array/tensor)
        :param y: Class label for the clip
        :param mode: 'train' or 'test' (to determine whether to use dropout or not
        :return: Mean loss for every frame
        """
        # target = T.tile(y, frames.shape[0])
        prediction_scores = self.get_output(frames, mode=mode)
        mean_score = prediction_scores.mean(axis=0)
        mean_score = mean_score.dimshuffle('x', 0)
        # prediction_counts = T.bincount(T.argmax(prediction_scores, axis=1))
        prediction = T.argmax(mean_score)
        y = y.dimshuffle('x', 0)
        # target = y.dimshuffle('x')
        loss = lasagne.objectives.categorical_crossentropy(predictions=mean_score, targets=y)
        return loss, prediction

    def predict(self, frames):
        pass
