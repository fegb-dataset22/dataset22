from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


def evaluate_model(model, X, y) -> Tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(X)
    errors = abs(predictions - y.values)
    return predictions, errors


def get_all_metrics(y: np.ndarray, predictions: np.ndarray, ) -> Dict[str, float]:
    regression_metrics = {
        "Mean Error": np.mean(predictions - y.values),
        "Mean Absolute Error": metrics.mean_absolute_error(y, predictions),
        "Mean Squared Error": metrics.mean_squared_error(y, predictions),
        "Root Mean Squared Error": metrics.mean_squared_error(y, predictions, squared=False),
        "Mean Absolute Percentage Error": metrics.mean_absolute_percentage_error(y, predictions),
        "Explained Variance Score": metrics.explained_variance_score(y, predictions),
        "Mean Squared Log Error": metrics.mean_squared_log_error(y, predictions),
        "Median Absolute Error": metrics.median_absolute_error(y, predictions),
        'R^2': metrics.r2_score(y, predictions)
    }

    return regression_metrics