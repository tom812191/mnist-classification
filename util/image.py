import numpy as np

from keras.preprocessing.image import ImageDataGenerator


def reshape_image(X, p=28):
    """
    Reshape the image from nxm to nxpxpx1 where n is the number of images and m is the total number of pixels such that
    m = p^2. The 4th dimension is for a single pixel intensity channel (4th dimension is needed for Keras CNN)

    :param X: np.array of nxm image data
    :param int p: sqrt of number of pixels
    :return: np.array nxpxpx1 image data
    """
    return X.reshape((-1, p, p, 1))


def normalize_image(X):
    """
    Normalize pixel values from 0..255 to 0..1

    :param X: np.array of nxpxpx1 image data
    :return: np.array of nxpxpx1 image data normalized
    """
    return X/255.0


def augment_data(X, y, factor=10):
    """
    Augment image data

    :param X: np.array of nxpxpx1 image data
    :param y: np.array of length n of labels

    :param factor: factor to increase image data by

    :return: X_aug, y_aug
    """

    datagen = ImageDataGenerator(
        rotation_range=10,  # Randomly rotate images up to 10 degrees
        zoom_range=0.1,  # Randomly zoom image up to 10%
        width_shift_range=0.1,  # Randomly shift left/right up to 10%
        height_shift_range=0.1,  # Randomly shift up/down up to 10%
    )

    X_out = np.copy(X)
    y_out = np.copy(y)

    total = 0
    batch_size = 10000
    for x_batch, y_batch in datagen.flow(X, y, batch_size=batch_size):
        # print('{} of {}'.format(total, len(X) * factor))
        X_out = np.concatenate((X_out, x_batch))
        y_out = np.concatenate((y_out, y_batch))
        total += batch_size
        if total >= len(X) * factor:
            break

    return X_out, y_out

