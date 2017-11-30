import xgboost as xgb

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import GridSearchCV

import numpy as np


def tune_xgb(X_train, y_train, use_evolution=True):
    """
    Optimize hyperparams for the XGBoost classifier

    :param X_train: Training images, np.array of shape nxpxpx1
    :param y_train: Training labels, np.array of shape n

    :param use_evolution: If true, use a genetic algorithm to tune hyperparams, else use an exhaustive grid search

    :return: The fitted model from the search (sklearn GridSearchCV object or EvolutionaryAlgorithmSearchCV object)
    """
    gbm = xgb.XGBClassifier(objective='multi:softprob')

    param_grid = {
        'max_depth': [5],
        'n_estimators': [10, 100, 500],
        'learning_rate': [0.1],
        # 'reg_alpha': [0, 0.2, 0.5, 1, 2],
        # 'reg_lambda': [0, 0.2, 0.5, 1, 2],
        # 'gamma': [0, 0.2, 0.5, 1, 2],
        # 'min_child_weight': [1, 2],

    }

    search_params = {
        'verbose': 10,
        'scoring': 'neg_log_loss',
        'cv': 2,
    }

    if use_evolution:
        search = EvolutionaryAlgorithmSearchCV(
            estimator=gbm,
            params=param_grid,
            population_size=10,
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=10,
            **search_params
        )
    else:
        search = GridSearchCV(gbm, param_grid=param_grid, **search_params)

    search.fit(X_train, y_train)

    return search


def fit_xgb(X_train, y_train, params):
    """
    Fit the XGBoost model with the given params

    :param X_train: Training images, np.array of shape nxpxpx1
    :param y_train: Training labels, np.array of shape n

    :param params: Hyperparams to use for the model

    :return: The fit XGBClassifier object
    """
    gbm = xgb.XGBClassifier(**params)
    gbm.fit(X_train, y_train)

    return gbm
