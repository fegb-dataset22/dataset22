import os.path
import pickle
from typing import Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data.data_helpers import get_original_data, get_region_counts_df, get_extended_df
from canova.src.canova.canova_source import canova
from root import ROOT_DIR


def build_df() -> pd.DataFrame:
    df = get_extended_df()
    df = df[["dG_pairing", "dG_folding", "meanR", "medianR", "maxR", "minR"]]
    return df


def canova_dependence(df: pd.DataFrame = None, file_path: str = None) -> Tuple[float, float]:
    df = build_df() if df is None else df
    meanR = df["meanR"]
    dG_pairing = df["dG_pairing"]
    dG_folding = df["dG_folding"]

    file_path = file_path if file_path else f"{ROOT_DIR}/data/canova_results.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            p_dG_pairing, p_dG_folding = pickle.load(f)

    else:
        p_dG_pairing = canova(dG_pairing, meanR)
        p_dG_folding = canova(dG_folding, meanR)

    print(f"p(dG_pairing) = {p_dG_pairing}")
    print(f"p(dG_folding) = {p_dG_folding}")
    return dG_pairing, dG_folding


def plot_correlations_heatmap(df: pd.DataFrame = None) -> None:
    df = build_df() if df is None else df
    df.drop(["maxR", "minR"], axis=1, inplace=True)
    corr = df.corr(method="spearman")

    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="vlag", annot=True)
    plt.show()


if __name__ == "__main__":
    built_df = build_df()
    plot_correlations_heatmap(built_df)
    canova_dependence(built_df)
