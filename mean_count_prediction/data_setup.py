import os.path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_helpers import get_extended_df
from root import ROOT_DIR


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    char_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    encoded = np.zeros((len(char_map.keys()), len(sequence)))
    for idx, char in enumerate(sequence):
        encoded[char_map[char], idx] = 1
    return encoded.flatten()


def inverse_one_hot_encode_sequence(encoded: np.ndarray) -> str:
    idx_map = {0: "A", 1: "C", 2: "G", 3: "U"}
    decoded = ""

    for i in range(np.shape(encoded)[0]):
        for j in range(np.shape(encoded)[1]):
            if encoded[i, j] == 1:
                decoded += idx_map[i]

    return decoded


def one_hot_encode_X(X: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame([one_hot_encode_sequence(sequence) for sequence in X["SeqID"]])
    X = X.drop(["SeqID"], axis=1)
    return encoded.join(X)


def split_data(df: pd.DataFrame = None, train_size=0.8, val_size=0.1, random_state=0) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_extended_df() if df is None else df
    X = df[["SeqID", "dG_pairing", "dG_folding"]]
    y = df["meanR"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size,
                                                        shuffle=True, random_state=random_state)

    test_ratio = 1 - train_size - val_size
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + val_size),
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def encoded_data(df: pd.DataFrame = None):
    df = get_extended_df() if df is None else df
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()

    X_train = one_hot_encode_X(X_train)
    X_val = one_hot_encode_X(X_val)
    X_test = one_hot_encode_X(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_encoded_data(X_train, X_val, X_test, y_train, y_val, y_test) -> None:
    X_train.to_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/X_train.csv", index=False)
    X_val.to_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/X_val.csv", index=False)
    X_test.to_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/X_test.csv", index=False)
    y_train.to_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/y_train.csv", index=False)
    y_val.to_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/y_val.csv", index=False)
    y_test.to_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/y_test.csv", index=False)


def get_encoded_data():
    try:
        X_train = pd.read_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/X_train.csv", )
        X_val = pd.read_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/X_val.csv", )
        X_test = pd.read_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/X_test.csv", )
        y_train = pd.read_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/y_train.csv", )
        y_val = pd.read_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/y_val.csv", )
        y_test = pd.read_csv(f"{ROOT_DIR}/mean_count_prediction/model_data/y_test.csv", )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except FileNotFoundError:
        data = encoded_data()
        save_encoded_data(*data)
        return data


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = encoded_data()
    print(f"{len(X_train)} Training Sequences ({len(X_train) / 260510})")
    print(f"{len(X_val)} Validation Sequences ({len(X_val) / 260510})")
    print(f"{len(X_test)} Testing Sequences ({len(X_test) / 260510})")

    save_encoded_data(X_train, X_val, X_test, y_train, y_val, y_test)
