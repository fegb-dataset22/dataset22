import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.model_storage import save_model, load_model
from weighted_mean_prediction.regression_performance import evaluate_model, get_all_metrics


def get_dG_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    return (X_train[["dG_pairing", "dG_folding"]].copy(), X_val[["dG_pairing", "dG_folding"]].copy(),
            X_test[["dG_pairing", "dG_folding"]].copy())


def fit_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
              file_path: str) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train, )
    save_model(model, file_path)
    return model


def normalise_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler().fit(X_test)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


if __name__ == "__main__":
    reg = LinearRegression()

    model_dir = f"{ROOT_DIR}/weighted_mean_prediction/linear_model/models"
    model_path = os.path.join(model_dir, "dg_lm.joblib")

    X_train, X_val, X_test, y_train, y_val, y_test = get_encoded_split_data()
    X_train, X_val, X_test = normalise_data(*get_dG_data(X_train, X_val, X_test))
    lm = load_model(model_path)
    lm = lm if lm is not None else fit_model(X_train, y_train["weighted_mean"], model_path)

    predictions, errors = evaluate_model(lm, X_test, y_test["weighted_mean"])
    print(get_all_metrics(y_test["weighted_mean"], predictions))
    plt.scatter(range(len(predictions[:100])), predictions[:100])
    plt.scatter(range(len(y_test["weighted_mean"][:100])), y_test["weighted_mean"][:100])
    plt.show()