import os
from typing import Callable, Optional

import joblib
import optuna
from optuna import Study


def load_study(file_path: Optional[str] = None,
               direction: str = "minimize") -> Study:
    if file_path is None or not os.path.exists(file_path):
        return optuna.create_study(direction=direction)

    return joblib.load(file_path)


def save_study(study: Study, file_path: str) -> None:
    joblib.dump(study, file_path)

