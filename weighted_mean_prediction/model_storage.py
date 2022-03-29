import os
from typing import Optional

import joblib
from sklearn.base import RegressorMixin


def load_model(file_path: str) -> Optional[RegressorMixin]:
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None


def save_model(model: RegressorMixin, file_path: str) -> None:
    joblib.dump(model, file_path)
