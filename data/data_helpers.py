from typing import Iterable, List

import pandas as pd

from root import ROOT_DIR


def get_original_data(file_path: str = None) -> pd.DataFrame:
    return pd.read_csv(f"{ROOT_DIR}/data/fepB_complete.csv") \
        if not file_path else pd.read_csv(file_path)


def get_weighted_mean_df(file_path: str = None) -> pd.DataFrame:
    return pd.read_csv(f"{ROOT_DIR}/data/fepB_weighted_mean.csv") \
        if not file_path else pd.read_csv(file_path)


def get_region_counts_df(df: pd.DataFrame = None):
    df = get_original_data() if df is None else df
    return df[[f"R{_}" for _ in range(1, 9)]]


def get_unique_letters(sequences: Iterable[str]) -> List[str]:
    return sorted(list({character for sequence in sequences
                        for character in sequence}))


def get_cumulative_cell_counts(df: pd.DataFrame = None) -> List[int]:
    df = get_original_data() if not df else df
    return [df[f"R{_}"].sum() for _ in range(1, 9)]


if __name__ == "__main__":
    d = get_original_data()
    print(d.head())
