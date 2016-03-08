__author__ = 'anushabala'
from argparse import ArgumentParser
from utils.models import AverageFrameModel, Outcome
from utils.ioutils import read_dataset
from utils.solver import FrameAverageSolver
import datetime

DEFAULT_MODEL_PATH = 'vgg16.pkl'


def param_summary(num_train, args):
    print "Training set size: %d" % num_train
    print "Number of frames per clip: %d" % args.max_frames
    print "Learning rate for output layer: %f" % args.output_lr
    print "Learning rate for tuning layers: %f" % args.tune_lr
    print "Batch size: %d" % args.batch_size
    print "Number of epochs: %d" % args.num_epochs


def main(args):
    dataset_json = args.json
    tuning_layers = args.tuning_layers
    vgg_path = args.vgg
    max_items = args.max_items
    max_frames = args.max_frames
    print "Loading VGGNet model"

    model = AverageFrameModel(vgg_path,
                              output_neurons=4,
                              tuning_layers=tuning_layers)

    start = datetime.datetime.now()

    data = read_dataset(dataset_json, sample_probability=0.5, mode='temporal',
                        max_items=max_items, max_frames=max_frames, mean_value=model.mean_bgr)
    print "Training data shape:", data["train_X"].shape

    end = datetime.datetime.now()
    print "Read data in %d seconds" % (end-start).seconds

    batch_size = min(args.batch_size, data["train_X"].shape[0])

    solver = FrameAverageSolver(model,
                                data["train_X"], data["train_y"],
                                val_X=data["val_X"], val_y=data["val_y"],
                                num_epochs=args.num_epochs,
                                batch_size=batch_size,
                                output_lr=args.output_lr,
                                tune_lr=args.tune_lr)

    solver.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='dataset.json',
                        help='Path to JSON file containing information about the location of the segmented clips and'
                             ' corresponding labels for each video. See sample_dataset.json for an example.')
    parser.add_argument('--vgg', type=str, default=DEFAULT_MODEL_PATH, help='Path to weights for pretrained VGG16 model (in .pkl format)')
    parser.add_argument('--tune', type=str, action='append', dest='tuning_layers', help='Name of layer(s) to tune weights for. This argument must be provided one for each layer separately. For example, python average_frame.py --tune fc7 --tune fc8 will tune the parameters for fc7 and fc8.')
    parser.add_argument('--max_items', type=int, default=10, help='Max items to load from the dataset')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames to load per clip')
    parser.add_argument('--output_lr', type=float, default=1e-4, help='Learning rate for final fully connected layer (output layer)')
    parser.add_argument('--tune_lr', type=float, default=1e-5, help='Learning rate for layers to be tuned')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for minibatch training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
    clargs = parser.parse_args()

    main(clargs)
