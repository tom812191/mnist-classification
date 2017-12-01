import numpy as np
from skimage import transform
import math


def classify(image, cnn):
    """
    Given an nxn image, classify the digit using the pre-trained models

    :param image: np.array of shape nxn
    :param cnn: CNN  model

    :return: label, probabilities
    """
    image = normalize_image(reshape_image(image.astype(float)))
    predictions = cnn.predict(image)
    return int(np.argmax(predictions)), predictions[0].tolist()


def reshape_image(image):
    """
    Reshape the numpy array to nxnx1

    :param image: np.array of shape nxn or 1d array of length n^2
    :return: np.array of shape nxnx1
    """
    shape = image.shape
    if len(shape) == 2:
        return image.reshape((shape[0], shape[1], 1))

    if len(shape) == 1:
        n = math.sqrt(shape[0])
        assert n == int(n)
        n = int(n)
        return image.reshape((n, n, 1))


def normalize_image(image, p=28):
    """
    Reshape image to 1xpxpx1 and convert all pixels to the range 0..1.
    Assumes image is square and size is >= 28x28

    :param image: np.array of shape nxnx1
    :param p: width and height of output image in pixels

    :return: np.array of shape 1xpxpx1 of the image
    """

    new_image = transform.resize(image, (p, p, 1))

    if image.max() > 1.01:
        new_image = new_image / 255.

    return np.array([new_image])
