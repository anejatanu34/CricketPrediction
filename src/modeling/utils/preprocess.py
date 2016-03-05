__author__ = 'anushabala'

import numpy as np
import skimage.transform


def preprocess_frames(frames, mean_value, size=(224,224)):
    preprocessed_frames = np.zeros_like(frames)
    raw_images = []
    i = 0
    for frame in frames:
        image = skimage.transform.resize(frame, size, preserve_range=True)
        raw_image = np.copy(frame).astype('uint8')
        raw_images.append(raw_image)
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)
        image = image[::-1, :, :]

        image = image - mean_value
        preprocessed_frames[i] = image

    return raw_images, preprocessed_frames
