from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import numpy as np


def evaluate_output(proba_predictions, y_test):
    """
    Evaluate the predictions based on accuracy, log loss, and confusion

    :param proba_predictions: np.array of probability of class predictions
    :param y_test: np.array of true y_test labels

    :return: dict with keys as the metrics and values as the results
    """
    predictions = np.argmax(proba_predictions, axis=1)

    return {
        'accuracy_score': accuracy_score(y_test, predictions),
        'log_loss': log_loss(y_test, proba_predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions),
    }