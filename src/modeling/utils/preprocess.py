from lasagne.utils import floatX

__author__ = 'anushabala'

import numpy as np
import skimage.transform


def preprocess_frames(frames, **kwargs):
    mean_value = kwargs['mean_value']
    size = kwargs.get('size', (224,224))
    preprocessed_frames = []
    raw_images = []
    for frame in frames:
        image = skimage.transform.resize(frame, size, preserve_range=True)
        raw_image = np.copy(frame).astype('uint8')
        raw_images.append(raw_image)
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)
        image = image[::-1, :, :]

        image = image - mean_value
        preprocessed_frames.append(floatX(image))

    return raw_images, preprocessed_frames
