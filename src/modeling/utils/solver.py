__author__ = 'anushabala'
import lasagne
import theano
from theano import tensor as T

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
        losses, updates = theano.scan(fn=lambda X_clip, output: self.model.clip_loss(X_clip, output),
                                      outputs_info=None,
                                      sequences=[input_var, output_var])

        loss = losses.mean()
        # todo compute accuracy
        output_layer = self.model.output_layer()
        # Get params for output layer and update using Adam
        params = output_layer.get_params(trainable=True)
        adam_update = lasagne.updates.adam(loss, params, learning_rate=self.output_lr)

        # Combine update expressions returned by theano.scan() with update expressions returned from the adam update
        for (key, value) in adam_update.iteritems():
            updates[key] = value

        # todo update layers that need to be finetuned
        self.train_function = theano.function([input_var, output_var], loss, updates=updates)

    def train(self):
        """
        Train the model for num_epochs with batches of size batch_size
        :return:
        """
        for i in xrange(self.num_epochs):
            print "Training epoch %d" % i
            for X_batch, y_batch in self.iterate_minibatches():
                print "Training batch 1"
                loss = self.train_function(X_batch, y_batch)
                print "Training loss: %f" % loss

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
