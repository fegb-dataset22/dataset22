import os.path
import sys

import pandas as pd
from optuna import Study

from hyperparameter_tuning.study_functions import load_study, save_study
from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.random_forest.rf_objective import RFObjective


def tune_rf_hyperparameters(X_train: pd.DataFrame, y_train: pd.DataFrame,
                            study: Study, file_path: str, n_trials: int = 20) -> None:
    try:
        study.optimize(RFObjective(X_train, y_train), n_trials=n_trials)
        save_study(study, file_path)
    except KeyboardInterrupt:
        save_study(study, file_path)
        best_params = study.best_params
        best_score = study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {best_params}\n")
        sys.exit()


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_encoded_split_data()
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    study_dir = f"{ROOT_DIR}/weighted_mean_prediction/random_forest/studies"
    study_name = f"rf_study1.joblib"
    study_path = os.path.join(study_dir, study_name)

    study = load_study(study_path)

    while True:
        tune_rf_hyperparameters(X_train, y_train["weighted_mean"], study, study_path, 5)
