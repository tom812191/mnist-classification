import luigi
import os
import numpy as np
import pandas as pd
import json

from config import DATA_ROOT

from jobs.preprocess import PreprocessRawData, AugmentData
from jobs.cnn import KerasCNNExtractFeatures
from estimators import xgb


class XGBModelSelection(luigi.Task):
    """
    Search the hyperparameter space to select the best XGBoost model
    """

    def requires(self):
        return [
            KerasCNNExtractFeatures(),
            AugmentData(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'pointers', 'XGBModelSelection.json'))

    def run(self):
        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        X_aug_train_features = np.load(paths['X_aug_train_features'])

        with self.input()[1].open('r') as path_file:
            paths = json.load(path_file)

        y_aug_train = np.load(paths['y_aug_train'])

        estimator = xgb.tune_xgb(X_aug_train_features, y_aug_train, use_evolution=True)

        paths = {
            'cv_results': os.path.join(DATA_ROOT, 'xgb_cv_results.csv'),
            'best_params': os.path.join(DATA_ROOT, 'xgb_best_params.json'),
        }

        with open(paths['cv_results'], 'w') as f:
            pd.DataFrame(estimator.cv_results_).to_csv(f, index=False)

        with open(paths['best_params'], 'w') as f:
            json.dump(estimator.best_params_, f)

        with self.output().open('w') as f:
            json.dump(paths, f)


class XGBFitModel(luigi.Task):
    """
    Fit the XGBoost model on all training data with optimal hyperparams and predict the output
    """

    def requires(self):
        return [
            KerasCNNExtractFeatures(),
            AugmentData(),
            XGBModelSelection(),
            PreprocessRawData(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_ROOT, 'pointers', 'XGBFitModel.json'))

    def run(self):
        with self.input()[0].open('r') as path_file:
            paths = json.load(path_file)

        X_aug_train_features = np.load(paths['X_aug_train_features'])

        with self.input()[1].open('r') as path_file:
            paths = json.load(path_file)

        y_aug_train = np.load(paths['y_aug_train'])

        with self.input()[2].open('r') as path_file:
            paths = json.load(path_file)

        params = json.load(paths['best_params'])

        estimator = xgb.fit_xgb(X_aug_train_features, y_aug_train, params)

        with self.input()[3].open('r') as path_file:
            paths = json.load(path_file)

        X_test = np.load(paths['X_test'])
        predictions = estimator.predict(X_test)
        proba_predictions = estimator.predict_proba(X_test)

        paths = {
            'predictions': os.path.join(DATA_ROOT, 'xgb_predictions.csv'),
            'proba_predictions': os.path.join(DATA_ROOT, 'xgb_proba_predictions.csv'),
        }

        np.save(paths['predictions'], predictions)
        np.save(paths['proba_predictions'], proba_predictions)

        with self.output().open('w') as f:
            json.dump(paths, f)
