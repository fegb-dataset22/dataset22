import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.linear_model.shared import normalise_data, fit_model
from weighted_mean_prediction.model_storage import load_model
from weighted_mean_prediction.regression_performance import evaluate_model, get_all_metrics, plot_residuals_histogram, \
    plot_QQ, plot_fitted, plot_fancy_fitted


def reshape_data(*X):
    return [x.values.reshape(-1, 1) for x in X]


if __name__ == "__main__":
    reg = LinearRegression()

    model_dir = f"{ROOT_DIR}/weighted_mean_prediction/linear_model/models"
    model_name = "pairing_lr.joblib"
    model_path = os.path.join(model_dir, model_name)

    X_train, X_val, X_test, y_train, y_val, y_test = get_encoded_split_data()
    # X_train, X_val, X_test  = get_dG_data(X_train, X_val, X_test)
    X_train, X_val, X_test = normalise_data(X_train, X_val, X_test)
    X_train, X_val, X_test = reshape_data(X_train["dG_pairing"], X_val["dG_pairing"], X_test["dG_pairing"])

    lm = load_model(model_path)
    lm = lm if lm is not None else fit_model(X_train, y_train["weighted_mean"], model_path)
    print(lm.coef_)
    predictions, errors = evaluate_model(lm, X_test, y_test["weighted_mean"])
    print(get_all_metrics(y_test["weighted_mean"], predictions))
    plt.scatter(X_test, predictions)
    plt.scatter(X_test, y_test)
    plt.show()

    plot_residuals_histogram(predictions, y_test["weighted_mean"])
    plot_QQ(predictions, y_test["weighted_mean"])
    plot_fancy_fitted(predictions, y_test["weighted_mean"])

    print(f"y = {lm.intercept_} + {lm.coef_[0]}x")
