import sys

__author__ = 'anushabala'
from argparse import ArgumentParser
from utils.models import LSTMGenerationModel
from utils.ioutils import read_dataset_commentary
from utils.gen_solver import EncoderDecoderSolver
import datetime
import json
import numpy as np
import lasagne

DEFAULT_MODEL_PATH = 'vgg16.pkl'


def param_summary(num_train, args):
    print "Training set size: %d" % num_train
    print "Validation size: %d" % args.val
    print "Number of frames per clip: %d" % args.max_frames
    print "Learning rate for output layer: %f" % args.output_lr
    print "Learning rate for tuning layers: %f" % args.tune_lr
    print "Layers to tune:", args.tune
    print "Batch size: %d" % args.batch_size
    print "Number of epochs: %d" % args.num_epochs
    print "Regularization factor: %f" % args.reg


def write_predictions(ids_file, predictions, outfile):
    ids = json.load(open(ids_file, 'r'))
    test_ids = ids["test"]
    out = open(outfile, 'w')

    for i in xrange(0, len(test_ids)):
        pred = "\"" + " ".join(predictions[i]) + "\""
        out.write("%s\t%s\n" % (test_ids[i], pred))

    out.close()


def main(args):
    dataset_json = args.json
    tuning_layers = []
    if args.tune:
        tuning_layers = args.tune.split(",")
        print tuning_layers
    vgg_path = args.vgg
    max_frames = args.max_frames

    model = LSTMGenerationModel(vgg_path, hidden_units=args.hidden, last_layer=args.last_layer,
                                max_frames=args.max_frames, seq_length=args.seq_length)

    print "Reading data"
    start = datetime.datetime.now()

    tvt_split = [args.train, args.val, args.test]
    data, word_to_idx, idx_to_word, vocab = read_dataset_commentary(dataset_json, sample_probability=0.5,
                                                                    mode='temporal',
                                                                    max_frames=max_frames, mean_value=model.mean_bgr,
                                                                    tvt_split=tvt_split, ids_file=args.ids,
                                                                    max_seq_length=args.seq_length,
                                                                    read_commentary=True)
    end = datetime.datetime.now()
    print "Read data in %d seconds" % (end - start).seconds

    train_shape = data["train_X"].shape
    data["train_X"] = data["train_X"].reshape((train_shape[0] * train_shape[1], train_shape[2],
                                               train_shape[3], train_shape[4]))
    test_shape = data["test_X"].shape
    data["test_X"] = data["test_X"].reshape((test_shape[0] * test_shape[1], test_shape[2],
                                             test_shape[3], test_shape[4]))
    val_shape = data["val_X"].shape
    data["val_X"] = data["val_X"].reshape((val_shape[0] * val_shape[1], val_shape[2],
                                           val_shape[3], val_shape[4]))

    print "Training data shape:", data["train_X"].shape
    print "Test data shape:", data["test_X"].shape
    print "Validation data shape:", data["val_X"].shape

    print len(vocab)
    model.initialize_model(len(vocab))
    # todo target shape needs to be changed probably?
    # print data["train_y"][0:2]
    # print data["train_mask"][0:2]
    # output = lasagne.layers.get_output(model.net['prob'],
    #                                    inputs={model.net["input"]: data["train_X"][0:10],
    #                                            model.net["mask_dec"]: data["train_mask"][0:2]})
    # print lasagne.layers.get_output(model.net['prob'],
    #                                 inputs={model.net["input"]: data["train_X"][0:10],
    #                                         model.net["mask_dec"]: data["train_mask"][0:2]}).eval().shape
    # loss = lasagne.objectives.categorical_crossentropy(output, data["train_y"][0:2].flatten())
    # print np.argmax(output.eval(), axis=1)
    # print loss.eval()
    # mask_flattened = data["train_mask"][0:2].flatten()
    # loss *= mask_flattened
    # print mask_flattened
    # print loss.eval()
    # sys.exit(0)
    print "---- Training parameters summary -----"
    param_summary(data["train_X"].shape[0], args)
    batch_size = min(args.batch_size, data["train_X"].shape[0]/args.max_frames)
    print "--------------------------------------"

    tuning_layers = ['lstm_decoder', 'lstm_encoder']
    if args.last_layer == 'fc6':
        tuning_layers.append('fc6')
    elif args.last_layer == 'fc7':
        tuning_layers.append('fc7')
        tuning_layers.append('fc6')
    solver = EncoderDecoderSolver(model,
                                  data["train_X"], data["train_y"],
                                  val_X=data["val_X"], val_y=data["val_y"],
                                  train_mask=data["train_mask"], val_mask=data["val_mask"],
                                  num_epochs=args.num_epochs,
                                  batch_size=batch_size,
                                  model_type='lstm_gen',
                                  output_lr=args.output_lr,
                                  tune_lr=args.output_lr,
                                  tuning_layers=tuning_layers,
                                  reg=args.reg,
                                  max_frames=args.max_frames,
                                  word_to_idx=word_to_idx,
                                  seq_length=args.seq_length,
                                  idx_to_word=idx_to_word)

    solver.train()
    test_predictions = solver.predict(data["train_X"], data["train_y"], data["train_mask"])
    # test_predictions, scores = solver.predict(data["test_X"], data["test_y"], data["test_mask"])
    write_predictions(args.ids, test_predictions, args.out)
    print "--------------------------------------"
    print "---- Training parameters summary -----"
    param_summary(data["train_X"].shape[0], args)

    print "---- Loss and accuracy history ----"
    print "Training loss history"
    print solver.train_loss_history
    print "Training accuracy history"
    print solver.train_acc_history
    print "Validation accuracy history"
    print solver.val_acc_history


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='dataset.json',
                        help='Path to JSON file containing information about the location of the segmented clips and'
                             ' corresponding labels for each video. See sample_dataset.json for an example.')

    parser.add_argument('--vgg', type=str, default=DEFAULT_MODEL_PATH, help='Path to weights for pretrained'
                                                                            ' VGG16 model (in .pkl format)')
    parser.add_argument('--tune', type=str, default=None, help='Name(s) of layer(s) to tune weights'
                                                               ' for (comma-separated).')
    parser.add_argument('--train', type=int, default=100, help='Number of clips in training set')
    parser.add_argument('--val', type=int, default=50, help='Number of clips in validation set')
    parser.add_argument('--test', type=int, default=50, help='Number of clips in test set')
    parser.add_argument('--max_items', type=int, default=10, help='Max items to load from the dataset')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames to load per clip')
    parser.add_argument('--output_lr', type=float, default=1e-4, help='Learning rate for final fully'
                                                                      ' connected layer (output layer)')
    parser.add_argument('--tune_lr', type=float, default=1e-5, help='Learning rate for layers to be tuned')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for minibatch training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--reg', type=float, default=1e-4, help='Regularization factor')
    parser.add_argument('--ids', type=str, default='clip_ids.txt', help='File to write ids of clips to.')
    parser.add_argument('--out', type=str, default='predictions.out', help='File to write predictions to.')
    parser.add_argument('--hidden', type=int, default=100, help='(LSTM only) number of hidden units')
    parser.add_argument('--seq_length', type=int, default=15, help='(text generation LSTM only) maximum length of '
                                                                   'input and output sequencs')
    parser.add_argument('--last_layer', type=str, default='fc7', choices=['fc6', 'fc7'],
                        help='Key for layer that feeds into LSTM layer')

    clargs = parser.parse_args()

    main(clargs)
