import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data.data_helpers import get_weighted_mean_df


def build_df() -> pd.DataFrame:
    df = get_weighted_mean_df()
    df = df[["dG_pairing", "dG_folding", "weighted_mean"]]
    return df


def plot_correlations_heatmap(df: pd.DataFrame = None, lower_only: bool = False,
                              cmap: str = "vlag") -> None:
    df = build_df() if df is None else df
    corr = df.corr(method="spearman")
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    if lower_only:
        sns.heatmap(corr, cmap=cmap, mask=mask, annot=False,)
    else:
        sns.heatmap(corr, cmap=cmap, annot=True)

    plt.show()


if __name__ == "__main__":
    built_df = build_df()
    plot_correlations_heatmap(built_df)
