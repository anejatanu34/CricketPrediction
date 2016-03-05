__author__ = 'anushabala'
from argparse import ArgumentParser
from utils.vgg_utils import CricketModel, Outcome
from utils.ioutils import read_dataset
from utils.preprocess import preprocess_frames
import numpy as np


DEFAULT_MODEL_PATH = 'vgg16.pkl'


def main(args):
    dataset = args.json
    tuning_layers = args.tuning_layers
    vgg_path = args.vgg

    model = CricketModel(vgg_path, output_neurons=4, tuning_layers=tuning_layers)
    X, y, raw_data = read_dataset(dataset, sample_probability=0.5, mean_value=model.mean_bgr)

    # visualize data
    print X.shape, y.shape, raw_data.shape


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='dataset.json',
                        help='Path to JSON file containing information about the location of the segmented clips and corresponding labels for each video. See sample_dataset.json for an example.')
    parser.add_argument('--vgg', type=str, default=DEFAULT_MODEL_PATH, help='Path to weights for pretrained VGG16 model (in .pkl format)')
    parser.add_argument('--tune', type=str, action='append', dest='tuning_layers', help='Name of layer(s) to tune weights for. This argument must be provided one for each layer separately. For example, python single_frame.py --tune fc7 --tune fc8 will tune the parameters for fc7 and fc8.')
    clargs = parser.parse_args()

    main(clargs)