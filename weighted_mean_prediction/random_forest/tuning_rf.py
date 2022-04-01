import os.path
import sys
from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt
from optuna import Study
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_contour, plot_edf, plot_optimization_history, plot_param_importances, \
    plot_slice

from hyperparameter_tuning.study_functions import load_study, save_study
from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.random_forest.rf_hyperparams import get_study1_hyperparams, get_study2_hyperparams
from weighted_mean_prediction.random_forest.rf_objective import RFObjective


def tune_rf_hyperparameters(X_train: pd.DataFrame, y_train: pd.DataFrame, get_params: Callable,
                            study: Study, file_path: str, n_trials: int = 20) -> None:
    try:
        study.optimize(RFObjective(X_train, y_train, get_params), n_trials=n_trials)
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
    study_name = f"rf_study2.joblib"
    study_path = os.path.join(study_dir, study_name)

    study = load_study(study_path)

    while True:
        tune_rf_hyperparameters(X_train, y_train["weighted_mean"], get_study2_hyperparams, study, study_path, 100)

        plot_edf(study)
        plot_contour(study)
        plot_optimization_history(study)
        plot_parallel_coordinate(study)
        plot_param_importances(study)
        plot_slice(study)
        plt.show()
