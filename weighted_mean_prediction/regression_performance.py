from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt, gridspec
import seaborn as sns
from sklearn import metrics

from data.visualisation import apply_style


def plot_rf_feature_importances(importances: np.ndarray) -> None:
    plt.title(f"Folding = {round(importances[-1], 2)}," f" Pairing = {round(importances[-2], 2)},"
              f" Sequence = {round(sum(importances[:-2]), 2)}")
    importances = pd.DataFrame(importances[:-2].view().reshape((4, 9)))

    sns.heatmap(np.round(importances, 3), annot=True, linecolor='b', cbar=True, cmap="vlag")
    plt.yticks([0, 1, 2, 3], ["A", "C", "G", "U"])
    plt.show()


def evaluate_model(model, X, y) -> Tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(X)
    errors = abs(predictions - y.values)
    return predictions, errors


def get_all_metrics(y: np.ndarray, predictions: np.ndarray, ) -> Dict[str, float]:
    regression_metrics = {
       # "Mean Error": np.mean(predictions - y.values),
        "Mean Absolute Error": metrics.mean_absolute_error(y, predictions),
        "Mean Squared Error": metrics.mean_squared_error(y, predictions),
        # "Root Mean Squared Error": metrics.mean_squared_error(y, predictions, squared=False),
        "Mean Absolute Percentage Error": metrics.mean_absolute_percentage_error(y, predictions),
        # "Explained Variance Score": metrics.explained_variance_score(y, predictions),
        # "Median Absolute Error": metrics.median_absolute_error(y, predictions),
        # 'R^2': metrics.r2_score(y, predictions)
    }

    return regression_metrics


def get_residuals(y_pred, y_test) -> np.ndarray:
    return y_pred - y_test


def plot_residuals_histogram(y_pred, y_test):
    apply_style()
    plt.hist(get_residuals(y_pred, y_test), density=True)
    plt.show()


def plot_QQ(y_pred, y_test):
    apply_style()
    sm.qqplot(get_residuals(y_pred, y_test), line='45')
    plt.show()


def plot_fitted(y_pred, y_test):
    apply_style()
    plt.scatter(y_pred, get_residuals(y_pred, y_test))
    plt.axhline(y=0, linestyle="--", color="k")
    plt.axvline(x=y_test.mean(), linestyle="--", color="r")
    plt.axvline(x=y_test.median(), linestyle="--", color="r")

    plt.show()


def plot_fancy_fitted(y_pred, y_test):
    x = y_pred
    y = get_residuals(y_pred, y_test)

    apply_style()
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_main.axhline(y=0, linestyle="--", color="k")
    ax_main.axvline(x=y_test.mean(), linestyle="--", color="r")
    ax_main.axvline(x=y_test.median(), linestyle="--", color="r")
    ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)

    # hb = ax_main.hexbin(x, y, bins='log')
    ax_main.scatter(x, y, alpha=0.05,)
    # ax_main.hist(y, orientation='horizontal', align='mid', )
    ax_main.set(xlabel="Target Variable", ylabel="Residuals")
    # cb = fig.colorbar(hb, ax=ax_main)
    # cb.set_label('log10(N)')
    ax_xDist.hist(x, align='mid', bins=100, density=True, alpha=0.5, label="Predicted")
    ax_xDist.hist(y_test, align='mid', bins=100, density=True, alpha=0.5, label="True")
    ax_xDist.legend()
    ax_xDist.set(ylabel='count')

    ax_yDist.hist(y, orientation='horizontal', align='mid', density=True)
    ax_yDist.set(xlabel='count')

    plt.show()

    plt.show()


def plot_lagged_residuals(y_pred, y_test):
    apply_style()
    res = get_residuals(y_pred, y_test)
    pass


if __name__ == "__main__":
    i = np.array([0.00786545, 0.00646138, 0.00831625, 0.0070642, 0.0065518, 0.00824827,
                  0.00717376, 0.00823789, 0.00783744, 0.01340055, 0.01737048, 0.01413202,
                  0.01706173, 0.0164007, 0.01496564, 0.01276124, 0.01389928, 0.00991164,
                  0.00927144, 0.00705245, 0.00841061, 0.00798325, 0.00864004, 0.00704123,
                  0.00912617, 0.01327073, 0.02230368, 0.00796729, 0.00712123, 0.00695599,
                  0.00744542, 0.00738225, 0.00677473, 0.00755621, 0.00678942, 0.006424,
                  0.52526618, 0.11955797])

    plot_rf_feature_importances(i)
