from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics


def plot_rf_feature_importances(importances: np.ndarray) -> None:
    plt.title(f"Folding = {round(importances[-1], 2)},"
              f" Pairing = {round(importances[-2], 2)}")
    importances = pd.DataFrame(importances[:-2].view().reshape((4, 9)))

    fig = plt.figure(figsize=(20, 15))
    ax1 = plt.subplot2grid((20, 20), (0, 0), colspan=19, rowspan=19)
    ax2 = plt.subplot2grid((20, 20), (19, 0), colspan=19, rowspan=1)
    ax3 = plt.subplot2grid((20, 20), (0, 19), colspan=1, rowspan=19)

    sns.heatmap(importances, ax=ax1, annot=True, linecolor='b', cbar=False, cmap="vlag")

    ax1.xaxis.tick_top()
    ax1.set_yticks([0, 1, 2, 3], ["A", "C", "G", "U"])

    sns.heatmap((pd.DataFrame(importances.sum(axis=0))).transpose(), ax=ax2, annot=True, cbar=False,
                xticklabels=False, yticklabels=False, cmap="vlag")
    sns.heatmap(pd.DataFrame(importances.sum(axis=1)), ax=ax3, annot=True, cbar=False, xticklabels=False,
                yticklabels=False, cmap="vlag")
    plt.show()


def evaluate_model(model, X, y) -> Tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(X)
    errors = abs(predictions - y.values)
    return predictions, errors


def get_all_metrics(y: np.ndarray, predictions: np.ndarray, ) -> Dict[str, float]:
    regression_metrics = {
        "Mean Error": np.mean(predictions - y.values),
        "Mean Absolute Error": metrics.mean_absolute_error(y, predictions),
        "Mean Squared Error": metrics.mean_squared_error(y, predictions),
        #"Root Mean Squared Error": metrics.mean_squared_error(y, predictions, squared=False),
        "Mean Absolute Percentage Error": metrics.mean_absolute_percentage_error(y, predictions),
        #"Explained Variance Score": metrics.explained_variance_score(y, predictions),
        #"Median Absolute Error": metrics.median_absolute_error(y, predictions),
        #'R^2': metrics.r2_score(y, predictions)
    }

    return regression_metrics


if __name__ == "__main__":
    i = np.array([0.00786545, 0.00646138, 0.00831625, 0.0070642, 0.0065518, 0.00824827,
                  0.00717376, 0.00823789, 0.00783744, 0.01340055, 0.01737048, 0.01413202,
                  0.01706173, 0.0164007, 0.01496564, 0.01276124, 0.01389928, 0.00991164,
                  0.00927144, 0.00705245, 0.00841061, 0.00798325, 0.00864004, 0.00704123,
                  0.00912617, 0.01327073, 0.02230368, 0.00796729, 0.00712123, 0.00695599,
                  0.00744542, 0.00738225, 0.00677473, 0.00755621, 0.00678942, 0.006424,
                  0.52526618, 0.11955797])

    plot_rf_feature_importances(i)
