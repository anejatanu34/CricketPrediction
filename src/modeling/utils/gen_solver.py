__author__ = 'anushabala'
from solver import Solver
from lasagne import regularization
import theano.tensor as T
import lasagne
import theano
import numpy as np
from nltk import bleu_score
import datetime


class EncoderDecoderSolver(Solver):
    def __init__(self, model,
                 train_X, train_y, val_X, val_y,
                 train_mask, val_mask,
                 **kwargs):

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.num_frames = kwargs.pop('max_frames', 5)
        self.seq_length = kwargs.pop('seq_length', 15)
        self.word_to_idx = kwargs.pop('word_to_idx')
        self.idx_to_word = kwargs.pop('idx_to_word')
        super(EncoderDecoderSolver, self).__init__(model, train_X, train_y, val_X, val_y, **kwargs)

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
            for X_batch, y_batch, batch_mask in self.iterate_minibatches():
                iters += 1
                if iters == 1:
                    initial_loss, initial_predictions = self.test_function(X_batch,y_batch, batch_mask)
                    initial_acc = self._compute_accuracy(initial_predictions, y_batch, batch_mask)
                    print "(%d/%d) Initial training loss: %f\tTraining accuracy:%2.4f" % (i, self.num_epochs, initial_loss, initial_acc)
                    self.train_loss_history.append((0, initial_loss))
                    self.train_acc_history.append((0, initial_acc))

                loss, predictions = self.train_function(X_batch, y_batch, batch_mask)
                acc = self._compute_accuracy(predictions, y_batch, batch_mask)

            print "(%d/%d) Training loss: %f\tTraining accuracy:%2.4f" % (i+1, self.num_epochs, loss, acc)
            self.train_loss_history.append((i+1, loss))
            self.train_acc_history.append((i+1, acc))

            if 0 < self.lr_decay < 1 and i % self.decay_after == 0 and i > 0:
                print "Decaying learning rates"
                self.output_lr *= self.lr_decay
                self.tuning_lr *= self.lr_decay

            if i % 5 == 0:
                self._check_val_accuracy()

        end = datetime.datetime.now()
        print "Training took %d seconds" % (end-start).seconds
        self._check_train_accuracy()
        if (self.num_epochs - 1) % 5 != 0:
            self._check_val_accuracy()

    def _init_train_fn(self):
        input_var = T.tensor4('input')
        mask_var = T.lmatrix('mask')
        output_var = T.lmatrix('targets')

        loss, predictions = self.model.loss(input_var, output_var, mask_var)
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
        self.train_function = theano.function([input_var, output_var, mask_var], [loss, predictions], updates=updates)

    def _init_test_fn(self):
        input_var = T.tensor4('input')
        output_var = T.lmatrix('output')
        mask_var = T.lmatrix('mask')

        # Compute losses by iterating over the input variable (a 5D tensor where each "row" represents a clip that
        # has some number of frames.
        loss, predictions,_ = self.model.loss(input_var, output_var, mask_var, mode='test')
        output_layer = self.model.layer('fc8')
        l2_penalty = regularization.regularize_layer_params(output_layer, regularization.l2) * self.reg * 0.5

        for layer_key in self.tuning_layers:
            layer = self.model.layer(layer_key)
            l2_penalty += regularization.regularize_layer_params(layer, regularization.l2) * self.reg * 0.5
        loss += l2_penalty

        self.test_function = theano.function([input_var, output_var, mask_var], [loss, predictions])

    def iterate_minibatches(self):
        """
        Iterate over minibatches in one epoch
        :return a single batch of the training data
        """
        num_train = self.train_X.shape[0]
        num_iterations_per_epoch = num_train/(self.batch_size * self.num_frames)
        indexes = np.arange(num_train/self.num_frames)
        for i in xrange(num_iterations_per_epoch):
            selected_idx = np.random.choice(indexes, self.batch_size, replace=False)
            mask = []
            for idx in selected_idx:
                mask.extend(np.arange(idx*self.num_frames,idx*self.num_frames+self.num_frames))

            yield self.train_X[mask], self.train_y[selected_idx], self.train_mask[selected_idx]

    def predict(self, X, y, mask):
        predictions = []
        num_test = X.shape[0]/self.num_frames
        start = 0
        test_acc = 0
        while start < num_test:
            end = min(start+self.batch_size, num_test)
            test_X = X[start*self.num_frames:end*self.num_frames]
            test_y = y[start:end]
            test_mask = mask[start:end]
            batch_loss, batch_predictions = self.test_function(test_X, test_y, test_mask)
            for prediction in batch_predictions:
                pred_words = [self.idx_to_word[i] for i in prediction]
                predictions.append(pred_words)
            # prediction_scores.extend(batch_scores)
            batch_acc = self._compute_accuracy(batch_predictions, test_y, test_mask)
            test_acc += batch_acc * self.batch_size

            start = end
        print "Accuracy on test set: %2.4f" % (test_acc/num_test)
        return predictions

    def _get_val_data(self):
        num_val = (self.val_X.shape[0]/self.num_frames)
        start = 0
        while start < num_val:
            end = min(start + self.batch_size, num_val)
            yield self.val_X[start*self.num_frames:end*self.num_frames], self.val_y[start:end], self.val_mask[start:end]
            start = end

    def _check_val_accuracy(self):
        val_loss = 0
        val_acc = 0
        num_val = self.val_X.shape[0]/self.num_frames
        for val_X_batch, val_y_batch, mask in self._get_val_data():
            batch_loss, batch_predictions = self.test_function(val_X_batch, val_y_batch, mask)
            val_loss += batch_loss * self.batch_size
            batch_acc = self._compute_accuracy(batch_predictions, val_y_batch, mask)
            val_acc += batch_acc * self.batch_size

        print "Validation loss: %f\tValidation accuracy:%2.2f" % (val_loss/num_val, val_acc/num_val)
        self.val_acc_history.append((self.num_epochs, val_acc))

    def _get_train_data(self):
        num_train = self.train_X.shape[0]/self.num_frames
        start = 0
        while start < num_train:
            end = min(start+ self.batch_size, num_train)
            yield self.train_X[start*self.num_frames:end*self.num_frames], self.train_y[start:end], self.train_mask[start:end]
            start = end

    def _check_train_accuracy(self):
        train_acc = 0
        num_train = self.train_X.shape[0]/self.num_frames
        for X_batch, y_batch, mask in self._get_train_data():
            batch_loss, train_predictions = self.test_function(X_batch, y_batch, mask)
            batch_acc = self._compute_accuracy(train_predictions, y_batch, mask)
            train_acc += batch_acc * self.batch_size

        print "Accuracy on complete training set: %2.4f" % (train_acc/num_train)
        self.val_acc_history.append((self.num_epochs, train_acc))

    def _compute_accuracy(self, predicted_y, true_y, mask):
        # compute bleu score
        # todo check if this really works, for now just return 0
        num_tested = true_y.shape[0]
        # total_score = 0
        references = []
        hypotheses = []
        # print "Predicted y", predicted_y
        # print "True y", true_y
        for i in xrange(num_tested):
            pred_idx = predicted_y[i]
            true_idx = true_y[i]
            current_mask = mask[i]
            pred_words = [self.idx_to_word[pred_idx[k]] for k in xrange(self.seq_length) if current_mask[k] != 0]
            # print pred_words
            true_words = [[self.idx_to_word[true_idx[k]] for k in xrange(self.seq_length) if current_mask[k] != 0]]
            # print true_words
            hypotheses.append(pred_words)
            references.append(true_words)

        score = bleu_score.corpus_bleu(references, hypotheses)
        return 0.0