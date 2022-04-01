import optuna
from typing import Dict


def get_study1_hyperparams(trial: optuna.trial) -> Dict[str, object]:
    _n_estimators = trial.suggest_int("n_estimators", 50, 200)
    _max_depth = trial.suggest_int("max_depth", 5, 20)
    _min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    _max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    return {
        "n_estimators": _n_estimators,
        "max_depth": _max_depth,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "max_features": _max_features,
    }


def get_study2_hyperparams(trial: optuna.trial) -> Dict[str, object]:
    _n_estimators = trial.suggest_int("n_estimators", 50, 200)
    _max_depth = trial.suggest_int("max_depth", 10, 40)
    _min_samples_split = 2
    _min_samples_leaf = 2
    _max_features = "sqrt"

    return {
        "n_estimators": _n_estimators,
        "max_depth": _max_depth,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "max_features": _max_features,
    }
