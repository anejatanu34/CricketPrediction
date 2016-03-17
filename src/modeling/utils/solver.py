__author__ = 'anushabala'
import lasagne
import theano
from theano import tensor as T
import numpy as np
import datetime
from lasagne import regularization
import sys

tensor5 = T.TensorType('floatX', (False,) * 5)


class Solver(object):
    """
    Solver for models.AverageFrameModel.
    todo: this can probably be generalized to all non-LSTM (and maybe even all LSTM) models,
    since the training function and loss function calls will probably stay the same?
    todo maybe create a generic Solver class instead?
    """
    def __init__(self, model,
                 train_X, train_y, val_X, val_y,
                 output_lr=1e-1, tune_lr=1e-3, lr_decay=0.95,
                 model_type='average',
                 num_epochs=1, batch_size=25, tuning_layers=[],
                 num_classes=4, reg=1e-4,
                 decay_after=1):
        """
        Create a new FrameAverageSolver instance
        :param model: Instance of the Model class (or a subclass of it) to train
        :param train_X: Training data (5D numpy array, of size (N,P,C,H,W))
        :param train_y: Labels for training data (1D vector, of size (N,))
        :param val_X: Validation data (5D numpy array, of size (N,P,C,H,W))
        :param val_y: Labels for validation data (1D vector, of size (N,))
        :param output_lr: Learning rate for output layer (fully connected layer)
        :param tune_lr: Learning rate for layers to be tuned
        :param num_epochs: Number of epochs to train for
        :param batch_size: Batch size for training
        :param tuning_layers: Keys of layers to tune
        """
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.output_lr = output_lr
        self.tuning_lr = tune_lr
        self.lr_decay = lr_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.reg = reg
        self.model_type = model_type
        self.tuning_layers = tuning_layers
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.decay_after = decay_after

        if self.model_type == 'late':
            self.tuning_layers = model.tuning_layers
            self.tuning_lr = self.output_lr
            print "Training layers: ", self.tuning_layers
            print "Learning rate", self.tuning_lr

        self._init_train_fn()
        self._init_test_fn()

    def _init_train_fn(self):
        self.train_function = lambda: None

    def _init_test_fn(self):
        self.test_function = lambda : None

    def train(self):
        """
        Train the model for num_epochs with batches of size batch_size
        :return:
        """

        iters = 0
        # compute initial validation loss and accuracy
        self._check_val_accuracy()

        print "Started model training"
        start = datetime.datetime.now()

        for i in xrange(self.num_epochs):
            loss = 0
            acc = 0
            for X_batch, y_batch in self.iterate_minibatches():
                iters += 1
                if iters == 1:
                    initial_loss, initial_predictions, scores = self.test_function(X_batch,y_batch)
                    initial_acc = self._compute_accuracy(initial_predictions, y_batch)
                    print "(%d/%d) Initial training loss: %f\tTraining accuracy:%2.4f" % (i, self.num_epochs, initial_loss, initial_acc)
                    self.train_loss_history.append((0, initial_loss))
                    self.train_acc_history.append((0, initial_acc))

                loss, predictions = self.train_function(X_batch, y_batch)
                acc = self._compute_accuracy(predictions, y_batch)

            print "(%d/%d) Training loss: %f\tTraining accuracy:%2.4f" % (i+1, self.num_epochs, loss, acc)
            self.train_loss_history.append((i+1, loss))
            self.train_acc_history.append((i+1, acc))

            if 0 < self.lr_decay < 1 and i % self.decay_after == 0 and i > 0:
                self.output_lr *= self.lr_decay
                self.tuning_lr *= self.lr_decay

            if i % 5 == 0:
                self._check_val_accuracy()

        end = datetime.datetime.now()
        print "Training took %d seconds" % (end-start).seconds
        self._check_train_accuracy()
        if (self.num_epochs - 1) % 5 != 0:
            self._check_val_accuracy()

    def _compute_accuracy(self, predicted_y, true_y):
        return np.array(predicted_y == true_y).mean()

    def _get_val_data(self):
        num_val = self.val_X.shape[0]
        start = 0
        while start < num_val:
            end = min(start+ self.batch_size, num_val)
            if self.model_type == 'late':
                yield np.take(self.val_X[start:end], indices=[0,-1],axis=1), self.val_y[start:end]
            else:
                yield self.val_X[start:end], self.val_y[start:end]
            start = end

    def _get_train_data(self):
        num_train = self.train_X.shape[0]
        start = 0
        while start < num_train:
            end = min(start+ self.batch_size, num_train)
            if self.model_type == 'late':
                yield np.take(self.train_X[start:end], indices=[0,-1],axis=1), self.train_y[start:end]
            else:
                yield self.train_X[start:end], self.train_y[start:end]
            start = end

    def _check_train_accuracy(self):
        train_acc = 0
        num_train = self.train_X.shape[0]
        for X_batch, y_batch in self._get_train_data():
            batch_loss, train_predictions, scores = self.test_function(X_batch, y_batch)
            batch_acc = self._compute_accuracy(train_predictions, y_batch)
            train_acc += batch_acc * self.batch_size

        print "Accuracy on complete training set: %2.4f" % (train_acc/num_train)
        self.val_acc_history.append((self.num_epochs, train_acc))

    def _check_val_accuracy(self):
        val_acc = 0
        num_val = self.val_X.shape[0]
        val_loss = 0
        for val_X_batch, val_y_batch in self._get_val_data():
            batch_loss, val_predictions, scores = self.test_function(val_X_batch, val_y_batch)
            val_loss += batch_loss * self.batch_size
            batch_acc = self._compute_accuracy(val_predictions, val_y_batch)
            val_acc += batch_acc * self.batch_size

        print "Validation loss: %f\tValidation accuracy:%2.4f" % (val_loss/num_val, val_acc/num_val)
        self.val_acc_history.append((self.num_epochs, val_acc))

    def predict(self, X, y):
        predictions = []
        prediction_scores = []
        num_test = X.shape[0]
        start = 0
        test_acc = 0
        while start < num_test:
            end = min(start+self.batch_size, num_test)
            test_X = X[start:end]
            if self.model_type == 'late':
                test_X = np.take(X[start:end], indices=[0, -1], axis=1)
            test_y = y[start:end]
            batch_loss, batch_predictions, batch_scores = self.test_function(test_X, test_y)
            predictions.extend(list(batch_predictions))
            prediction_scores.extend(batch_scores)
            batch_acc = self._compute_accuracy(batch_predictions, test_y)
            test_acc += batch_acc * self.batch_size

            start = end
        print "Accuracy on test set: %2.4f" % (test_acc/num_test)
        return predictions, prediction_scores

    def iterate_minibatches(self):
        """
        Iterate over minibatches in one epoch
        :return a single batch of the training data
        """
        num_train = self.train_X.shape[0]
        num_iterations_per_epoch = num_train/self.batch_size
        indexes = np.arange(num_train)
        for i in xrange(num_iterations_per_epoch):
            mask = np.random.choice(indexes, self.batch_size, replace=False)
            if self.model_type == 'average':
                yield self.train_X[mask], self.train_y[mask]
            elif self.model_type == 'late':
                yield np.take(self.train_X[mask], indices=[0,-1], axis=1), self.train_y[mask]


class CNNSolver(Solver):

    """
    Solver for models.AverageFrameModel.
    todo: this can probably be generalized to all non-LSTM (and maybe even all LSTM) models,
    since the training function and loss function calls will probably stay the same?
    todo maybe create a generic Solver class instead?
    """

    def _init_train_fn(self):
        """
        Initialize Theano function to compute loss and update weights using Adam for a single epoch and minibatch.
        """
        input_var = tensor5('input')
        output_var = T.lvector('output')
        one_hot = T.extra_ops.to_one_hot(output_var, self.num_classes, dtype='int64')

        # output_one_hot = T.extra_ops.to_one_hot(output_var, self.num_classes, dtype='int64')
        # Compute losses by iterating over the input variable (a 5D tensor where each "row" represents a clip that
        # has some number of frames.
        [losses, predictions], updates = theano.scan(fn=lambda X_clip, output: self.model.clip_loss(X_clip, output),
                                                     outputs_info=None,
                                                     sequences=[input_var, one_hot])

        loss = losses.mean()

        output_layer = self.model.layer('fc8')
        l2_penalty = regularization.regularize_layer_params(output_layer, regularization.l2) * self.reg * 0.5
        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            l2_penalty += regularization.regularize_layer_params(layer, regularization.l2) * self.reg * 0.5
        loss += l2_penalty

        # Get params for output layer and update using Adam
        params = output_layer.get_params(trainable=True)
        adam_update = lasagne.updates.adam(loss, params, learning_rate=self.output_lr)

        # Combine update expressions returned by theano.scan() with update expressions returned from the adam update
        updates.update(adam_update)
        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            layer_params = layer.get_params(trainable=True)
            layer_adam_updates = lasagne.updates.adam(loss, layer_params, learning_rate=self.tuning_lr)
            updates.update(layer_adam_updates)
        self.train_function = theano.function([input_var, output_var], [loss, predictions], updates=updates)

    def _init_test_fn(self):
        input_var = tensor5('input')
        output_var = T.lvector('output')
        one_hot = T.extra_ops.to_one_hot(output_var, self.num_classes, dtype='int64')

        # Compute losses by iterating over the input variable (a 5D tensor where each "row" represents a clip that
        # has some number of frames.
        [losses, predictions, scores], updates = theano.scan(fn=lambda X_clip, output: self.model.clip_loss(X_clip, output, mode='test'),
                                                             outputs_info=None,
                                                             sequences=[input_var, one_hot])
        loss = losses.mean()
        output_layer = self.model.layer('fc8')
        l2_penalty = regularization.regularize_layer_params(output_layer, regularization.l2) * self.reg * 0.5
        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            l2_penalty += regularization.regularize_layer_params(layer, regularization.l2) * self.reg * 0.5
        loss += l2_penalty

        self.test_function = theano.function([input_var, output_var], [loss, predictions, scores], updates=updates)


class LSTMSolver(Solver):
    def __init__(self, model,
                 train_X, train_y, val_X, val_y,
                 **kwargs):

        self.seq_length = kwargs.pop('max_frames', 5)
        super(LSTMSolver, self).__init__(model, train_X, train_y, val_X, val_y, **kwargs)

    def _init_train_fn(self):
        input_var = T.tensor4('input')
        output_var = T.lvector('targets')
        one_hot = T.extra_ops.to_one_hot(output_var, self.num_classes, dtype='int64')

        loss, predictions = self.model.loss(input_var, one_hot)
        output_layer = self.model.layer('fc8')
        l2_penalty = regularization.regularize_layer_params(output_layer, regularization.l2) * self.reg * 0.5

        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            l2_penalty += regularization.regularize_layer_params(layer, regularization.l2) * self.reg * 0.5
        loss += l2_penalty

        # Get params for output layer and update using Adam
        params = output_layer.get_params(trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=self.output_lr)

        # Combine update expressions returned by theano.scan() with update expressions returned from the adam update
        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            layer_params = layer.get_params(trainable=True)
            layer_adam_updates = lasagne.updates.adam(loss, layer_params, learning_rate=self.tuning_lr)
            updates.update(layer_adam_updates)
        self.train_function = theano.function([input_var, output_var], [loss, predictions], updates=updates)

    def _init_test_fn(self):
        input_var = T.tensor4('input')
        output_var = T.lvector('output')
        one_hot = T.extra_ops.to_one_hot(output_var, self.num_classes, dtype='int64')

        # Compute losses by iterating over the input variable (a 5D tensor where each "row" represents a clip that
        # has some number of frames.
        loss, predictions, scores = self.model.loss(input_var, one_hot, mode='test')
        output_layer = self.model.layer('fc8')
        l2_penalty = regularization.regularize_layer_params(output_layer, regularization.l2) * self.reg * 0.5

        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            l2_penalty += regularization.regularize_layer_params(layer, regularization.l2) * self.reg * 0.5
        loss += l2_penalty

        self.test_function = theano.function([input_var, output_var], [loss, predictions, scores])

    def iterate_minibatches(self):
        """
        Iterate over minibatches in one epoch
        :return a single batch of the training data
        """
        num_train = self.train_X.shape[0]
        num_iterations_per_epoch = num_train/(self.batch_size * self.seq_length)
        indexes = np.arange(num_train/self.seq_length)
        for i in xrange(num_iterations_per_epoch):
            selected_idx = np.random.choice(indexes, self.batch_size, replace=False)
            mask = []
            for idx in selected_idx:
                mask.extend(np.arange(idx*self.seq_length,idx*self.seq_length+self.seq_length))

            yield self.train_X[mask], self.train_y[selected_idx]

    def predict(self, X, y):
        predictions = []
        prediction_scores = []
        num_test = X.shape[0]/self.seq_length
        start = 0
        test_acc = 0
        while start < num_test:
            end = min(start+self.batch_size, num_test)
            test_X = X[start*self.seq_length:end*self.seq_length]
            test_y = y[start:end]
            batch_loss, batch_predictions, batch_scores = self.test_function(test_X, test_y)
            predictions.extend(list(batch_predictions))
            prediction_scores.extend(batch_scores)
            batch_acc = self._compute_accuracy(batch_predictions, test_y)
            test_acc += batch_acc * self.batch_size

            start = end
        print "Accuracy on test set: %2.4f" % (test_acc/num_test)
        return predictions, prediction_scores

    def _get_val_data(self):
        num_val = (self.val_X.shape[0]/self.seq_length)
        start = 0
        while start < num_val:
            end = min(start + self.batch_size, num_val)
            yield self.val_X[start*self.seq_length:end*self.seq_length], self.val_y[start:end]
            start = end

    def _check_val_accuracy(self):
        val_loss = 0
        val_acc = 0
        num_val = self.val_X.shape[0]/self.seq_length
        for val_X_batch, val_y_batch in self._get_val_data():
            batch_loss, batch_predictions, batch_scores = self.test_function(val_X_batch, val_y_batch)
            val_loss += batch_loss * self.batch_size
            batch_acc = self._compute_accuracy(batch_predictions, val_y_batch)
            val_acc += batch_acc * self.batch_size

        print "Validation loss: %f\tValidation accuracy:%2.2f" % (val_loss/num_val, val_acc/num_val)
        self.val_acc_history.append((self.num_epochs, val_acc))

    def _get_train_data(self):
        num_train = self.train_X.shape[0]/self.seq_length
        start = 0
        while start < num_train:
            end = min(start+ self.batch_size, num_train)
            yield self.train_X[start*self.seq_length:end*self.seq_length], self.train_y[start:end]
            start = end

    def _check_train_accuracy(self):
        train_acc = 0
        num_train = self.train_X.shape[0]/self.seq_length
        for X_batch, y_batch in self._get_train_data():
            batch_loss, train_predictions, scores = self.test_function(X_batch, y_batch)
            batch_acc = self._compute_accuracy(train_predictions, y_batch)
            train_acc += batch_acc * self.batch_size

        print "Accuracy on complete training set: %2.4f" % (train_acc/num_train)
        self.val_acc_history.append((self.num_epochs, train_acc))
