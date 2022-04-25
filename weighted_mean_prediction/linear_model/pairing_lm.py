import os
import sys

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.linear_model.shared import normalise_data, fit_model
from weighted_mean_prediction.model_storage import load_model
from weighted_mean_prediction.regression_performance import evaluate_model, get_all_metrics, plot_residuals_histogram, \
    plot_QQ, plot_fitted, plot_fancy_fitted


def add_quadratic_pairing(*X):
    for x in X:
        x["x^2"] = x["dG_pairing"].pow(2)


if __name__ == "__main__":
    reg = LinearRegression()

    model_dir = f"{ROOT_DIR}/weighted_mean_prediction/linear_model/models"
    model_name = "pairing_lm.joblib"
    model_path = os.path.join(model_dir, model_name)

    X_train, X_val, X_test, y_train, y_val, y_test = get_encoded_split_data()
    X_train, X_val, X_test = normalise_data(X_train, X_val, X_test)
    add_quadratic_pairing(X_train, X_val, X_test)

    lm = load_model(model_path)
    lm = lm if lm is not None else fit_model(X_train[["dG_pairing", "x^2"]], y_train["weighted_mean"], model_path)
    predictions, errors = evaluate_model(lm, X_test[["dG_pairing", "x^2"]], y_test["weighted_mean"])
    print(get_all_metrics(y_test["weighted_mean"], predictions))
    plt.scatter(X_test["dG_pairing"], y_test)
    plt.scatter(X_test["dG_pairing"], predictions)
    plt.show()

    plot_residuals_histogram(predictions, y_test["weighted_mean"])
    plot_QQ(predictions, y_test["weighted_mean"])
    plot_fitted(predictions, y_test["weighted_mean"])
    plot_fancy_fitted(predictions, y_test["weighted_mean"])

    print(f"y = {lm.intercept_} + {lm.coef_[0]}x + {lm.coef_[1]}x^2")
