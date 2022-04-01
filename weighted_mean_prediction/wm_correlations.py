import matplotlib.pyplot as plt
import seaborn as sns

from data.data_helpers import get_weighted_mean_df
from data.dg_correlations import plot_correlations_heatmap
from weighted_mean_prediction.data_setup import one_hot_encode_df

if __name__ == "__main__":
    df = get_weighted_mean_df().drop([f"R{_}" for _ in range(1, 9)], axis=1)
    df = one_hot_encode_df(df)
    plot_correlations_heatmap(df, lower_only=True)

    corr_series = df.corrwith(df["weighted_mean"]).drop(["weighted_mean"])

    print(corr_series["dG_pairing"])
    print(corr_series["dG_folding"])
    sns.heatmap(corr_series.drop(["dG_pairing", "dG_folding"]).values.round(2).reshape(4, 9),
                xticklabels="auto", yticklabels=["A", "C", "G", "U"], annot=True,
                cmap="coolwarm")
    plt.xlabel("Sequence Index")
    plt.show()
