import xgboost as xgb

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import GridSearchCV

import numpy as np


def tune_xgb(X_train, y_train, use_evolution=True):
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

    param_grid = {
        'max_depth': np.linspace(1, 9, 5).astype(int),
        'n_estimators': np.linspace(100, 1500, 5).astype(int),
        'learning_rate': [0.0005, 0.005, 0.05, 0.1, 0.3],
        'reg_alpha': [0, 0.2, 0.5, 1, 2],
        'reg_lambda': [0, 0.2, 0.5, 1, 2],
        'gamma': [0, 0.2, 0.5, 1, 2],
        'min_child_weight': np.linspace(0.1, 1.5, 5),

    }

    search_params = {
        'verbose': 10,
        'scoring': 'neg_log_loss',
        'cv': 5,
    }

    if use_evolution:
        search = EvolutionaryAlgorithmSearchCV(
            estimator=gbm,
            params=param_grid,
            population_size=20,
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=20,
            **search_params
        )
    else:
        search = GridSearchCV(gbm, param_grid=param_grid, **search_params)

    search.fit(X_train, y_train)

    return search


def fit_xgb(X_train, y_train, params):
    gbm = xgb.XGBClassifier(**params)
    gbm.fit(X_train, y_train)

    return gbm
