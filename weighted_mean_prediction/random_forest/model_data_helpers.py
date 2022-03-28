import os
from typing import Optional

import joblib
from sklearn.ensemble import RandomForestRegressor


def load_rf_model(file_path: str) -> Optional[RandomForestRegressor]:
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None


def save_model(model: RandomForestRegressor, file_path: str) -> None:
    joblib.dump(model, file_path)
