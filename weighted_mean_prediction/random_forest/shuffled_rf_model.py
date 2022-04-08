import os.path

import pandas as pd
from sklearn import metrics

from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.model_storage import load_model
from weighted_mean_prediction.random_forest.base_rf_model import train_base_model


def shuffle(df: pd.Series):
    return df.sample(frac=1)


if __name__ == "__main__":
    study_dir = f"{ROOT_DIR}/weighted_mean_prediction/random_forest/studies"
    study_name = f"rf_study_base_shuffled.joblib"
    study_path = os.path.join(study_dir, study_name)
    model_name = "base_rf_shuffled.joblib"
    model_path = f"{ROOT_DIR}/weighted_mean_prediction/random_forest/models/{model_name}"

    X_train, X_val, X_test, y_train, y_val, y_test = get_encoded_split_data()
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    y_train = shuffle(y_train)
    y_test = shuffle(y_test)

    rf = load_model(model_path)
    rf = rf if rf is not None else train_base_model(X_train, y_train["weighted_mean"], model_path)

    predictions = rf.predict(X_test)
    mape = metrics.mean_absolute_percentage_error(predictions, y_test)
    mse = metrics.mean_squared_error(predictions, y_test)
    print("\nMAPE = ", mape)
    print("MSE = ", mse)

    """
    MAPE =  0.432119796606408
    MSE =  2.9531582226998623
    """