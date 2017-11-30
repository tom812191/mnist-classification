import numpy as np
from skimage import transform, color
from scipy import misc


def load_png(path):
    return color.rgb2gray(misc.imread(path)) * 255.


def classify(image, cnn_feat, xgb):
    """
    Given an nxn image, classify the digit using the pre-trained models

    :param image: np.array of shape nxn
    :param cnn_feat: CNN Feature extractor model
    :param xgb: XGBoost model

    :return: label, probability
    """
    image = normalize_image(reshape_image(image))
    features = cnn_feat.predict(image)
    label, probabilities = xgb.predict(features)[0], xgb.predict_proba(features)

    return label, probabilities.max()


def classify_cnn(image, cnn):
    image = normalize_image(reshape_image(image))
    probs = cnn.predict(image)[0]

    return probs.argmax(), probs.max(), image


def reshape_image(image):
    """
    Reshape the numpy array from nxn to nxnx1

    :param image: np.array of shape nxn
    :return: np.array of shape nxnx1
    """
    shape = image.shape
    return image.reshape((shape[0], shape[1], 1))


def normalize_image(image, p=28):
    """
    Reshape image to 1xpxpx1 and convert all pixels to the range 0..1.
    Assumes image is square and size is >= 28x28

    :param image: np.array of shape nxnx1
    :param p: width and height of output image in pixels

    :return: np.array of shape 1xpxpx1 of the image
    """

    new_image = transform.resize(image, (p, p, 1))
    new_image = new_image / 255.

    return np.array([new_image])
