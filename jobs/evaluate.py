import luigi
import os
import numpy as np
import json
import pickle

from keras.models import load_model

from config import DATA_ROOT

from jobs.preprocess import PreprocessRawData
from jobs.cnn import KerasCNNFitModel

from util import evaluate


class EvaluateCNN(luigi.Task):
    """
    Calculate metrics on the final fit CNN model
    """

    def requires(self):
        return [
            KerasCNNFitModel(),
            PreprocessRawData(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'evaluation.p'))

    def run(self):
        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        model = load_model(paths['model'])

        with self.input()[1].open('r') as path_file:
            paths = json.load(path_file)

        X_test = np.load(paths['X_test'])
        y_test = np.load(paths['y_test'])

        proba_predictions = model.predict(X_test)

        evaluation = evaluate.evaluate_output(proba_predictions, y_test)

        with open(self.output().path, 'wb') as f:
            pickle.dump(evaluation, f)
