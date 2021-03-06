import luigi
import os
import numpy as np
import json
import pickle

from keras.models import load_model

from config import DATA_ROOT, CNN_CONFIG

from jobs.preprocess import AugmentData, PreprocessRawData
from estimators import cnn


class KerasCNNFitModel(luigi.Task):
    """
    Fir the Keras Convolutional Neural Network
    """

    def requires(self):
        return [
            AugmentData(),
            PreprocessRawData(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'pointers', 'KerasCNNFitModel.json'))

    def run(self):

        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        X_aug_train = np.load(paths['X_aug_train'])
        y_aug_train = np.load(paths['y_aug_train'])

        with self.input()[1].open('r') as path_file:
            paths = json.load(path_file)

        X_test = np.load(paths['X_test'])
        y_test = np.load(paths['y_test'])

        history, model = cnn.train_model(X_aug_train, y_aug_train, X_test, y_test,
                                         epochs=CNN_CONFIG['epochs'], batch_size=CNN_CONFIG['batch_size'])

        paths = {
            'history': os.path.join(DATA_ROOT, 'cnn_history.p'),
            'model': os.path.join(DATA_ROOT, 'cnn.h5'),
        }

        with open(paths['history'], 'wb') as history_file:
            pickle.dump(history.history, history_file)

        model.save(paths['model'])

        with self.output().open('w') as f:
            json.dump(paths, f)


class KerasCNNExtractFeatureModel(luigi.Task):
    """
    Save the feature model from the full model
    """

    def requires(self):
        return [
            KerasCNNFitModel(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'cnn-feature-model.h5'))

    def run(self):

        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        model = load_model(paths['model'])
        feature_model = cnn.get_feature_extraction_model(model)

        feature_model.save(self.output().path)


class KerasCNNExtractFeatures(luigi.Task):
    """
    Extract features from the trained convolutional neural network
    """

    def requires(self):
        return [
            KerasCNNExtractFeatureModel(),
            AugmentData(),
            PreprocessRawData(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'pointers', 'KerasCNNExtractFeatures.json'))

    def run(self):

        feature_model = load_model(self.input()[0].path)

        with self.input()[1].open('r') as path_file:
            paths = json.load(path_file)

        X_aug_train = np.load(paths['X_aug_train'])

        with self.input()[2].open('r') as path_file:
            paths = json.load(path_file)

        X_test = np.load(paths['X_test'])

        X_aug_train_features = feature_model.predict(X_aug_train)
        X_test_features = feature_model.predict(X_test)

        paths = {
            'X_aug_train_features': os.path.join(DATA_ROOT, 'X_aug_train_features.npy'),
            'X_test_features': os.path.join(DATA_ROOT, 'X_test_features.npy'),
        }

        np.save(paths['X_aug_train_features'], X_aug_train_features)
        np.save(paths['X_test_features'], X_test_features)

        with self.output().open('w') as f:
            json.dump(paths, f)
