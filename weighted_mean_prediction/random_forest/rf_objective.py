import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


class RFObjective:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame,
                 n_splits: int = 5):
        self.X = X
        self.y = y

        self.n_splits = n_splits

    def __call__(self, trial):
        _n_estimators = trial.suggest_int("n_estimators", 50, 200)
        _max_depth = trial.suggest_int("max_depth", 5, 20)
        _min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
        _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
        #_max_features = trial.suggest_discrete_uniform("max_features", 1, len(self.X.columns), 1)
        _max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
        rf = RandomForestRegressor(
            max_depth=_max_depth,
            min_samples_split=_min_samp_split,
            min_samples_leaf=_min_samples_leaf,
            max_features=_max_features,
            n_estimators=_n_estimators,
            n_jobs=-1,
            random_state=42,
        )

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(rf, self.X, self.y, cv=cv,
                                 scoring="neg_root_mean_squared_error")
        return - scores.mean()
