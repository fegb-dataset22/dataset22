from typing import Dict, Callable

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


class RFObjective:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, get_rf_params: Callable,
                 n_splits: int = 10):
        self.X = X
        self.y = y
        self.get_rf_params = get_rf_params
        self.n_splits = n_splits

    def __call__(self, trial):
        rf = RandomForestRegressor(**self.get_rf_params(trial), random_state=42,)

        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(rf, self.X, self.y, cv=cv,
                                 scoring="neg_root_mean_squared_error")

        return - scores.mean()
