__author__ = 'anushabala'
import lasagne
import theano
from theano import tensor as T
import numpy as np
import datetime

tensor5 = T.TensorType('float64', (False,) * 5)


class FrameAverageSolver(object):
    """
    Solver for models.AverageFrameModel.
    todo: this can probably be generalized to all non-LSTM (and maybe even all LSTM) models,
    since the training function and loss function calls will probably stay the same?
    todo maybe create a generic Solver class instead?
    """
    def __init__(self, model, train_X, train_y, val_X, val_y, output_lr=1e-2, tune_lr=1e-3, num_epochs=1,
                 batch_size=25):
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.output_lr = output_lr
        self.tune_lr = tune_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self._init_train_fn()

    def _init_train_fn(self):
        """
        Initialize Theano function to compute loss and update weights using Adam for a single epoch and minibatch.
        """
        input_var = tensor5('input')
        output_var = T.lvector('output')
        # Compute losses by iterating over the input variable (a 5D tensor where each "row" represents a clip that
        # has some number of frames.
        [losses, predictions], updates = theano.scan(fn=lambda X_clip, output: self.model.clip_loss(X_clip, output),
                                                     outputs_info=None,
                                                     sequences=[input_var, output_var])

        loss = losses.mean()
        output_layer = self.model.output_layer()
        # Get params for output layer and update using Adam
        params = output_layer.get_params(trainable=True)
        adam_update = lasagne.updates.adam(loss, params, learning_rate=self.output_lr)

        # Combine update expressions returned by theano.scan() with update expressions returned from the adam update
        for (key, value) in adam_update.iteritems():
            updates[key] = value

        # todo update layers that need to be finetuned
        self.train_function = theano.function([input_var, output_var], [loss, predictions], updates=updates)

    def _init_test_fn(self):
        input_var = tensor5('test_input')
        output_var = T.lvector('test_output')
        [losses, predictions], updates = theano.scan(fn=lambda X_clip, output: self.model.clip_loss(X_clip, output, mode='test'),
                                                     outputs_info=None,
                                                     sequences=[input_var, output_var])
        loss = losses.mean()

        self.test_function = theano.function([input_var, output_var], [loss, predictions], updates=updates)

    def train(self):
        """
        Train the model for num_epochs with batches of size batch_size
        :return:
        """
        print "Started model training"
        start = datetime.datetime.now()
        iterations_per_epoch = max(self.train_y.shape[0] / self.batch_size, 1)
        num_iterations = iterations_per_epoch * self.num_epochs

        iters = 0
        for i in xrange(self.num_epochs):
            loss = 0
            acc = 0
            for X_batch, y_batch in self.iterate_minibatches():
                i += 1
                loss, predictions = self.train_function(X_batch, y_batch)
                acc = self._compute_accuracy(predictions, y_batch)
            print "(%d/%d) Training loss: %f\tTraining accuracy:%2.2f" % (iters, num_iterations, loss, acc)

            # check validation accuracy every 5 epochs
            # todo maybe change this to every X epochs depending on # of epochs, or maybe make this parameterizable
            if i % 5 == 0:
                test_loss, test_predictions = self.test_function(self.val_X, self.val_y)
                test_acc = self._compute_accuracy(test_predictions, self.val_y)
                print "\tTest loss: %f\tTest accuracy:%2.2f" % (test_loss, test_acc)

        end = datetime.datetime.now()
        print "Training took %d seconds" % (end-start).seconds

    def _compute_accuracy(self, predicted_y, true_y):
        return np.array(predicted_y == true_y).mean()

    def iterate_minibatches(self):
        """
        Iterate over minibatches in one epoch
        :return a single batch of the training data
        """
        ctr = 0
        while ctr < self.train_y.shape[0]:
            end = min(ctr + self.batch_size, self.train_X.shape[0])
            yield self.train_X[ctr:end], self.train_y[ctr:end]
            ctr += self.batch_size
