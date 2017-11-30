import os

DATA_ROOT = '/Users/tom/Projects/Portfolio/data/mnist-classification'

DATA_TRAIN = os.path.join(DATA_ROOT, 'mnist_train.csv')
DATA_TEST = os.path.join(DATA_ROOT, 'mnist_test.csv')

CNN_CONFIG = {
    'epochs': 30,
    'batch_size': 100,
    'cnn_model_path': os.path.join(DATA_ROOT, 'cnn.h5')
}

PREPROCESS_CONFIG = {
    'data_augmentation_factor': 2,
}