from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_helpers import get_original_data
from root import ROOT_DIR
from weighted_mean_prediction.data_setup import one_hot_encode_df, split_data

global REGIONS
REGIONS = [f"R{_}" for _ in range(1, 9)]


def get_normalised_region_counts(df: pd.DataFrame = None) -> pd.DataFrame:
    df = get_original_data() if df is None else df
    df = df.copy()
    df[REGIONS] = df[REGIONS].div(df[REGIONS].sum(axis=1), axis=0)
    return df


def save_encoded_data(X_train, X_val, X_test, y_train, y_val, y_test) -> None:
    X_train.to_csv(f"{ROOT_DIR}/region_count_prediction/model_data/X_train.csv", index=False)
    X_val.to_csv(f"{ROOT_DIR}/region_count_prediction/model_data/X_val.csv", index=False)
    X_test.to_csv(f"{ROOT_DIR}/region_count_prediction/model_data/X_test.csv", index=False)
    y_train.to_csv(f"{ROOT_DIR}/region_count_prediction/model_data/y_train.csv", index=False)
    y_val.to_csv(f"{ROOT_DIR}/region_count_prediction/model_data/y_val.csv", index=False)
    y_test.to_csv(f"{ROOT_DIR}/region_count_prediction/model_data/y_test.csv", index=False)


def get_encoded_split_data():
    try:
        X_train = pd.read_csv(f"{ROOT_DIR}/region_count_prediction/model_data/X_train.csv", )
        X_val = pd.read_csv(f"{ROOT_DIR}/region_count_prediction/model_data/X_val.csv", )
        X_test = pd.read_csv(f"{ROOT_DIR}/region_count_prediction/model_data/X_test.csv", )
        y_train = pd.read_csv(f"{ROOT_DIR}/region_count_prediction/model_data/y_train.csv", )
        y_val = pd.read_csv(f"{ROOT_DIR}/region_count_prediction/model_data/y_val.csv", )
        y_test = pd.read_csv(f"{ROOT_DIR}/region_count_prediction/model_data/y_test.csv", )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except FileNotFoundError:
        data = split_data()
        save_encoded_data(*data)
        return data


def split_data(df: pd.DataFrame = None, train_size=0.8, val_size=0.1, random_state=0) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_original_data() if df is None else df

    X = df.drop(REGIONS, axis=1)
    y = df[REGIONS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size,
                                                        shuffle=True, random_state=random_state)

    test_ratio = 1 - train_size - val_size
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + val_size),
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = get_normalised_region_counts()
    df = one_hot_encode_df(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print(f"{len(X_train)} Training Sequences ({len(X_train) / 260510})")
    print(f"{len(X_val)} Validation Sequences ({len(X_val) / 260510})")
    print(f"{len(X_test)} Testing Sequences ({len(X_test) / 260510})")

    save_encoded_data(X_train, X_val, X_test, y_train, y_val, y_test)
