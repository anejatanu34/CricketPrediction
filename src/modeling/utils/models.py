__author__ = 'anushabala'

from .vgg16 import *
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

    def get_output(self, input_images, mode='train', layer_name=output_key):
        if mode == 'test':
            return lasagne.layers.get_output(self.layer(layer_name), input_images, deterministic=True)
        else:
            return lasagne.layers.get_output(self.layer(layer_name), input_images, deterministic=False)


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
        if mode == 'train':
            return loss, prediction
        else:
            return loss, prediction, prediction_scores

    def predict(self, frames):
        pass


class LateFusionModel(Model):
    merge_fc_layer = 'fc6'
    reshape_layer = 'rshp5_4'

    def __init__(self, path, mean_key='mean value',
                 weights_key='param values', output_neurons=4, class_labels=Outcome.class_labels()):
        self.net = build_late_fusion_model(output_neurons)
        model = pickle.load(open(path))
        self.labels = class_labels
        self.mean_bgr = np.reshape(model[mean_key], (3,1,1))
        self.model_weights = model[weights_key]
        self._set_model_params()

    def _set_model_params(self):
        """
        Set params according to VGG16 pretrained weights for all layers except output layer
        """
        last_layer = self.net[self.reshape_layer].input_layer
        lasagne.layers.set_all_param_values(last_layer, self.model_weights[:-6])

    def clip_loss(self, X, y, mode='train'):
        """
        Get the loss for a single clip by computing the combined loss of the first and last frame of the clip
        :param X: All the frames in a single clip
        :param y: Output label for clip
        :param mode: 'train' or 'test'
        :return:
        """
        prediction_scores = self.get_output(X, mode=mode)
        # prediction_scores = prediction_scores.dimshuffle((1,))
        prediction = T.argmax(prediction_scores, axis=1)
        # prediction_scores = prediction_scores.dimshuffle('x', 0)
        y = y.dimshuffle('x', 0)
        loss = lasagne.objectives.categorical_crossentropy(prediction_scores, y)
        if mode == 'train':
            return loss, prediction[0]
        else:
            return loss, prediction[0], prediction_scores


class LSTMModel(Model):
    merge_fc_layer = 'fc6'

    def __init__(self, path, mean_key='mean value',
                 weights_key='param values', output_neurons=4, class_labels=Outcome.class_labels(),
                 vocab_size=4000, max_frames=5, hidden_units=100, last_layer='fc7'):
        self.last_layer_key = last_layer
        self.net = build_lstm_classification_model(output_neurons, max_frames=max_frames,
                                                   hidden_units=hidden_units, last_layer=last_layer)
        model = pickle.load(open(path))
        self.labels = class_labels
        self.mean_bgr = np.reshape(model[mean_key], (3,1,1))
        self.model_weights = model[weights_key]
        self.vocab_size = vocab_size
        self._set_model_params()

    def _set_model_params(self):
        """
        Set params according to VGG16 pretrained weights for all layers except output layer
        """
        last_layer = self.net[self.last_layer_key]
        if self.last_layer_key == 'fc7':
            lasagne.layers.set_all_param_values(last_layer, self.model_weights[:-2])
        elif self.last_layer_key == 'fc6':
            lasagne.layers.set_all_param_values(last_layer, self.model_weights[:-4])
        else:
            raise ValueError('Last LSTM model layer must be one of fc7 or fc6')

    def loss(self, X, y, mode='train'):
        prediction_scores = self.get_output(X, mode=mode)
        loss = lasagne.objectives.categorical_crossentropy(prediction_scores, y)
        predictions = T.argmax(prediction_scores, axis=1)

        if mode == 'train':
            return loss.mean(), predictions
        return loss.mean(), predictions, prediction_scores


class LSTMGenerationModel(Model):
    merge_fc_layer = 'fc6'

    def __init__(self, path, mean_key='mean value',
                 weights_key='param values', output_neurons=4, class_labels=Outcome.class_labels(),
                 max_frames=5, hidden_units=100, last_layer='fc7', seq_length=15):
        self.last_layer_key = last_layer
        self.output_neurons = output_neurons
        self.max_frames = max_frames
        self.hidden_units = hidden_units
        self.last_layer = last_layer
        model = pickle.load(open(path))
        self.labels = class_labels
        self.mean_bgr = np.reshape(model[mean_key], (3,1,1))
        self.model_weights = model[weights_key]
        self.vocab_size = 2000
        self.seq_length = seq_length

    def initialize_model(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.net = build_lstm_commentary_model(self.output_neurons, max_frames=self.max_frames,
                                               hidden_units=self.hidden_units, last_layer=self.last_layer,
                                               vocab_size=self.vocab_size, seq_length=self.seq_length)
        self._set_model_params()

    def _set_model_params(self):
        """
        Set params according to VGG16 pretrained weights for all layers except output layer
        """
        last_layer = self.net[self.last_layer_key]
        if self.last_layer_key == 'fc7':
            lasagne.layers.set_all_param_values(last_layer, self.model_weights[:-2])
        elif self.last_layer_key == 'fc6':
            lasagne.layers.set_all_param_values(last_layer, self.model_weights[:-4])
        else:
            raise ValueError('Last LSTM model layer must be one of fc7 or fc6')

    def loss(self, X, y, mask, mode='train'):
        if mode == 'train':
            scores = lasagne.layers.get_output(self.net['prob'],
                                               inputs={self.net["input"]:X,
                                                       self.net["mask_dec"]:mask},
                                               deterministic=True)
        else:
            scores = lasagne.layers.get_output(self.net['prob'],
                                               inputs={self.net["input"]:X,
                                                       self.net["mask_dec"]:mask},
                                               deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(scores, T.flatten(y))
        loss = loss * T.flatten(mask)
        predictions_flattened = T.argmax(scores, axis=1)
        predictions = T.reshape(predictions_flattened, newshape=(-1, self.seq_length))

        if mode =='train':
            return loss.mean(), predictions
        else:
            return loss.mean(), predictions, scores
