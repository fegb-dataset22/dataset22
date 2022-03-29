import os.path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from hyperparameter_tuning.study_functions import load_study
from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.model_storage import load_model, save_model
from weighted_mean_prediction.regression_performance import plot_rf_feature_importances


def is_same_params(study_params: Dict[str, object],
                   model_params: Dict[str, object]) -> bool:
    return all([study_params[p] == model_params[p] for p in study_params.keys()])


def train_tuned_model(model_params: Dict[str, object], X_train: pd.DataFrame, y_train: pd.DataFrame,
                      file_path: str) -> RandomForestRegressor:
    model = RandomForestRegressor(**model_params, random_state=0)
    model.fit(X_train, y_train)
    save_model(model, file_path)
    return model


if __name__ == "__main__":
    study_dir = f"{ROOT_DIR}/weighted_mean_prediction/random_forest/studies"
    study_name = f"rf_study1.joblib"
    study_path = os.path.join(study_dir, study_name)
    model_name = "rf1.joblib"
    model_path = f"{ROOT_DIR}/weighted_mean_prediction/random_forest/models/{model_name}"

    X_train, X_val, X_test, y_train, y_val, y_test = get_encoded_split_data()
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    study = load_study(study_path)

    rf: RandomForestRegressor = load_model(model_path)

    if rf is None:
        rf = train_tuned_model(study.best_params, X_train, y_train["weighted_mean"], model_path)
    else:
        if not is_same_params(study.best_params, rf.get_params()):
            rf = train_tuned_model(study.best_params, X_train, y_train["weighted_mean"], model_path)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(train_acc, test_acc)

    for idx, importance in enumerate(rf.feature_importances_):
        print(f"{X_train.columns[idx]} : {importance}")

    plot_rf_feature_importances(rf.feature_importances_)

    predictions = rf.predict(X_test)
    mape = metrics.mean_absolute_percentage_error(predictions, y_test)
    mse = metrics.mean_squared_error(predictions, y_test)
    print("\nMAPE = ", mape)
    print("MSE = ", mse)

    plt.scatter(range(len(predictions[:100])), predictions[:100])
    plt.scatter(range(len(y_test[:100])), y_test[:100])
    plt.show()