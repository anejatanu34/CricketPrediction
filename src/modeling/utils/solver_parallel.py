__author__ = 'anushabala'

import lasagne
import theano
from theano import tensor as T
import numpy as np
import datetime
import math
from multiprocessing import Queue, Process

tensor5 = T.TensorType('floatX', (False,) * 5)

# todo implement masks


class FrameAverageSolver(object):
    """
    Solver for models.AverageFrameModel.
    todo: this can probably be generalized to all non-LSTM (and maybe even all LSTM) models,
    since the training function and loss function calls will probably stay the same?
    todo maybe create a generic Solver class instead?
    """
    def __init__(self, model, train_X, train_y, val_X, val_y, output_lr=1e-2, tune_lr=1e-3, num_epochs=1,
                 batch_size=25, tuning_layers=[], num_proc=1):
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
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.tuning_layers = tuning_layers
        self.num_proc = num_proc
        self._init_train_fn()
        self._init_test_fn()
        self.Q_in = [Queue()] * self.num_proc
        self.Q_out = [Queue()] * self.num_proc

    def _init_train_fn(self):
        """
        Initialize Theano function to compute loss and update weights using Adam for a single epoch and minibatch.
        """
        loss = T.dscalar()
        output_layer = self.model.output_layer()
        # Get params for output layer and update using Adam
        params = output_layer.get_params(trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=self.output_lr)

        # Combine update expressions returned by theano.scan() with update expressions returned from the adam update
        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            layer_params = layer.get_params(trainable=True)
            layer_adam_updates = lasagne.updates.adam(loss, layer_params, learning_rate=self.tuning_lr)
            updates.update(layer_adam_updates)

        self.train_function = theano.function([loss], updates=updates)

    def _init_test_fn(self):
        input_var = tensor5('test_input')
        output_var = T.lvector('test_output')
        [losses, predictions], updates = theano.scan(fn=lambda X_clip, output: self.model.clip_loss(X_clip, output, mode='test'),
                                                     outputs_info=None,
                                                     sequences=[input_var, output_var])
        loss = losses.mean()

        self.test_function = theano.function([input_var, output_var], [loss, predictions], updates=updates)

    def _compute_loss_grad_exp(self, q_in, q_out):
        """
        Get the loss and gradient for this part of the mini-batch.
        """
        while True:
            if not q_in.empty():
                batch_submask = q_in.get()
                if type(batch_submask) is str and batch_submask=='quit':
                    break

                X_batch = self.X_train[batch_submask]
                y_batch = self.y_train[batch_submask]
                loss, predictions = self.model.clip_loss(X_batch, y_batch)

                q_out.put([batch_submask, loss, predictions])
        return

    def train(self):
        """
        Train the model for num_epochs with batches of size batch_size
        :return:
        """
        print "Started model training"
        start = datetime.datetime.now()
        iterations_per_epoch = max(self.train_y.shape[0] / self.batch_size, 1)
        num_iterations = iterations_per_epoch * self.num_epochs

        # Initialize Processes
        num_train = self.train_X.shape[0]
        if num_train < self.batch_size:
            self.batch_size = num_train


        Procs = {}
        for i in range(self.num_proc):
            Procs[i] = Process(target=self._compute_loss_grad_exp, args=(self.Q_in[i], self.Q_out[i]))
            Procs[i].start()

        block_size = 1.0*self.batch_size/self.num_proc
        iters = 0
        for i in xrange(self.num_epochs):
            acc = 0
            loss = 0
            for batch_mask in self.iterate_minibatches():
                loss = 0
                for i in range(self.num_proc):
                    mask_start = math.ceil(i*block_size)
                    mask_end = min(num_train, math.ceil((i+1)*block_size))
                    batch_submask = batch_mask[mask_start:mask_end]
                    self.Q_in[i].put(batch_submask)

                # Multi-Process Collecting
                count = 0
                all_predictions = np.zeros_like(self.train_y[batch_mask])
                while count!=self.num_proc:
                    for i in range(self.num_proc):
                        if not self.Q_out[i].empty():
                            count +=1
                            batch_submask, subloss, batch_predictions = self.Q_out[i].get() # not to confuse with technical "subgradient"
                            loss += len(batch_submask) * subloss
                            all_predictions[batch_submask] = batch_predictions
                iters += 1

                acc = self._compute_accuracy(all_predictions, self.train_y[batch_mask])
                print "(%d/%d) Training loss: %f\tTraining accuracy:%2.2f" % (iters, num_iterations, loss, acc)

                self.train_function(self.train_X[batch_mask], self.train_y[batch_mask], loss)



            # check validation accuracy every 5 epochs
            # todo maybe change this to every X epochs depending on # of epochs, or maybe make this parameterizable
            if i % 5 == 0:
                val_loss, test_predictions = self.test_function(self.val_X, self.val_y)
                val_acc = self._compute_accuracy(test_predictions, self.val_y)
                print "\tTest loss: %f\tTest accuracy:%2.2f" % (val_loss, val_acc)

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
            yield np.arange(ctr, end)
            ctr += self.batch_size
