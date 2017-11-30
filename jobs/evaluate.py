import luigi
import os
import numpy as np
import json

from config import DATA_ROOT

from jobs.preprocess import PreprocessRawData
from jobs.xgb import XGBFitModel

from util import evaluate


class Evaluate(luigi.Task):
    """
    Search the hyperparameter space to select the best XGBoost model
    """

    def requires(self):
        return [
            XGBFitModel(),
            PreprocessRawData(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'evaluation.json'))

    def run(self):
        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        predictions = np.load(paths['predictions'])
        proba_predictions = np.load(paths['proba_predictions'])

        with self.input()[1].open('r') as path_file:
            paths = json.load(path_file)

        y_test = np.load(paths['y_test'])

        evaluation = evaluate.evaluate_output(predictions, proba_predictions, y_test)

        with self.output().open('w') as f:
            json.dump(evaluation, f)
