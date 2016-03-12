__author__ = 'anushabala'
from argparse import ArgumentParser
from utils.models import AverageFrameModel, LateFusionModel, LSTMModel
from utils.ioutils import read_dataset_tvt
from utils.solver import CNNSolver, LSTMSolver
import datetime
import json
import numpy as np

DEFAULT_MODEL_PATH = 'vgg16.pkl'
model_types = ['average', 'late', 'lstm']


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


def write_predictions(model_type, ids_file, predictions, scores, outfile):
    ids = json.load(open(ids_file, 'r'))
    test_ids = ids["test"]
    out = open(outfile, 'w')

    for i in xrange(0, len(test_ids)):
        norm_scores = np.array(scores[i])
        if model_type == 'average':
            scores_str = str([str(x) for x in norm_scores])
        else:
            scores_str = "\t".join([str(x) for x in list(norm_scores)])
        out.write("%s\t%s\t%s\n" % (test_ids[i], predictions[i], scores_str))

    out.close()


def main(args):
    dataset_json = args.json
    tuning_layers = []
    if args.tune:
        tuning_layers = args.tune.split(",")
        print tuning_layers
    vgg_path = args.vgg
    max_frames = args.max_frames
    print "Loading VGGNet model. Model type: %s" % args.model_type

    if args.model_type == 'average':
        model = AverageFrameModel(vgg_path,
                                  output_neurons=4,
                                  tuning_layers=tuning_layers)
    elif args.model_type == 'late':
        model = LateFusionModel(vgg_path, output_neurons=4)
    elif args.model_type == 'lstm':
        model = LSTMModel(vgg_path, hidden_units=args.hidden, last_layer=args.last_layer,
                          max_frames=args.max_frames)
    else:
        raise ValueError("Model type must be one of 'average' and 'late'")

    print "Reading data"
    start = datetime.datetime.now()

    tvt_split = [args.train, args.val, args.test]
    data = read_dataset_tvt(dataset_json, sample_probability=0.5, mode='temporal', max_frames=max_frames,
                            mean_value=model.mean_bgr, tvt_split=tvt_split, ids_file=args.ids)

    end = datetime.datetime.now()
    print "Read data in %d seconds" % (end - start).seconds

    if args.model_type == 'lstm':
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
    
    print "---- Training parameters summary -----"
    param_summary(data["train_X"].shape[0], args)
    batch_size = min(args.batch_size, data["train_X"].shape[0])
    print "--------------------------------------"
    if args.model_type == 'lstm':
        tuning_layers = ['lstm']
        if args.last_layer == 'fc6':
            tuning_layers.append('fc6')
        elif args.last_layer == 'fc7':
            tuning_layers.append('fc7')
            tuning_layers.append('fc6')
        solver = LSTMSolver(model,
                            data["train_X"], data["train_y"],
                            val_X=data["val_X"], val_y=data["val_y"],
                            model_type=args.model_type,
                            num_epochs=args.num_epochs,
                            batch_size=batch_size,
                            output_lr=args.output_lr,
                            tune_lr=args.output_lr,
                            tuning_layers=tuning_layers,
                            reg=args.reg,
                            max_frames=args.max_frames)
    else:
        solver = CNNSolver(model,
                           data["train_X"], data["train_y"],
                           val_X=data["val_X"], val_y=data["val_y"],
                           model_type=args.model_type,
                           num_epochs=args.num_epochs,
                           batch_size=batch_size,
                           output_lr=args.output_lr,
                           tune_lr=args.tune_lr,
                           tuning_layers=tuning_layers,
                           reg=args.reg)

    solver.train()
    test_predictions, scores = solver.predict(data["test_X"], data["test_y"])
    write_predictions(args.model_type, args.ids, test_predictions, scores, args.out)
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
    parser.add_argument('--model_type', type=str, default='average', choices=model_types)
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
    parser.add_argument('--last_layer', type=str, default='fc7', choices=['fc6', 'fc7'],
                        help='Key for layer that feeds into LSTM layer')

    clargs = parser.parse_args()

    main(clargs)
