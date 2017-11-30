import luigi
import pandas as pd
import numpy as np
import os
import json

from config import DATA_TRAIN, DATA_TEST, DATA_ROOT

from util import image


class AugmentData(luigi.Task):
    """
    Augment the training set
    """
    def requires(self):
        return [PreprocessRawData()]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'pointers', 'AugmentData.json'))

    def run(self):
        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        X_train = np.load(paths['X_train'])
        y_train = np.load(paths['y_train'])

        X_aug_train, y_aug_train = image.augment_data(X_train, y_train, factor=10)

        paths = {
            'X_aug_train': os.path.join(DATA_ROOT, 'X_aug_train.npy'),
            'y_aug_train': os.path.join(DATA_ROOT, 'y_aug_train.npy'),
        }

        np.save(paths['X_aug_train'], X_aug_train)
        np.save(paths['y_aug_train'], y_aug_train)

        with self.output().open('w') as f:
            json.dump(paths, f)


class PreprocessRawData(luigi.Task):
    """
    Take input data into numpy arrays
    """

    def requires(self):
        return [RawTrainData(), RawTestData()]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'pointers', 'PreprocessRawData.json'))

    def run(self):
        with self.input()[0].open('r') as training_data_file:
            df_train = pd.read_csv(training_data_file, header=None)

        with self.input()[1].open('r') as test_data_file:
            df_test = pd.read_csv(test_data_file, header=None)

        X_train = image.normalize_image(image.reshape_image(df_train.iloc[:, 1:].values, p=28))
        y_train = df_train.iloc[:, 0].values

        X_test = image.normalize_image(image.reshape_image(df_test.iloc[:, 1:].values, p=28))
        y_test = df_test.iloc[:, 0].values

        paths = {
            'X_train': os.path.join(DATA_ROOT, 'X_train.npy'),
            'y_train': os.path.join(DATA_ROOT, 'y_train.npy'),
            'X_test': os.path.join(DATA_ROOT, 'X_test.npy'),
            'y_test': os.path.join(DATA_ROOT, 'y_test.npy'),
        }

        np.save(paths['X_train'], X_train)
        np.save(paths['y_train'], y_train)
        np.save(paths['X_test'], X_test)
        np.save(paths['y_test'], y_test)

        with self.output().open('w') as f:
            json.dump(paths, f)


class RawTrainData(luigi.task.ExternalTask):
    """
    Dummy task for training input data
    """
    def output(self):
        return luigi.LocalTarget(DATA_TRAIN)


class RawTestData(luigi.task.ExternalTask):
    """
    Dummy task for test input data
    """
    def output(self):
        return luigi.LocalTarget(DATA_TEST)
