from typing import Iterable, List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

from data.data_helpers import get_unique_letters, get_cumulative_cell_counts, get_original_data, get_region_counts_df


def apply_style(style: str = "seaborn-bright") -> None:
    plt.style.use(style)


def generate_heatmap_data(sequences: Iterable[str], ) -> np.ndarray:
    data = np.zeros((len(get_unique_letters(sequences)), len(sequences[0])))
    unique_letters = list(get_unique_letters(sequences))

    for sequence in sequences:
        for index, char in enumerate(sequence):
            data[unique_letters.index(char)][index] += 1

    return data


def plot_heatmap() -> None:
    sequences = get_original_data()["SeqID"]
    data = normalize(generate_heatmap_data(sequences), axis=0, norm='l1')

    fig, ax = plt.subplots()
    im = ax.imshow(data, )

    ax.set_xticks(np.arange(np.shape(data)[1]), np.arange(np.shape(data)[1]))
    ax.set_yticks(np.arange(np.shape(data)[0]), get_unique_letters(sequences))

    # Loop over data dimensions and create text annotations.
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            text = ax.text(j, i, round(data[i, j], 3),
                           ha="center", va="center", color="w")
    plt.show()


def plot_cumulative_cell_distribution() -> None:
    x = range(1, 9)
    counts = [count / sum(get_cumulative_cell_counts())
              for count in get_cumulative_cell_counts()]

    apply_style()
    plt.bar(x, counts)
    plt.ylabel("Frequency")
    plt.xlabel("Fluorescent Region")
    plt.xticks(x, [f"R{_}" for _ in x])
    plt.show()


def plot_dG_feature(feature: str = "pairing",
                    metric: str = "mean") -> None:
    valid_features = ["pairing", "folding"]
    valid_metrics = ["mean", "median"]
    if feature.lower() not in valid_features:
        raise ValueError(f"Not a valid argument select from {valid_features}")
    if metric.lower() not in valid_metrics:
        raise ValueError(f"Not a valid argument select from {valid_metrics}")

    df = get_original_data()
    counts_df = get_region_counts_df(df)

    mean_df = counts_df.mean(axis=1)
    median_df = counts_df.median(axis=1)

    apply_style()
    if metric == "mean":
        plt.scatter(df[f"dG_{feature}"], mean_df)
        plt.ylim([0, 1000])  # 1 Very extreme outlier
    elif metric == "median":
        plt.scatter(df[f"dG_{feature}"], median_df, alpha=0.5)
        plt.ylim([0, 100])

    plt.ylabel(f"{metric} Cell Count")
    plt.xlabel(f"dG_{feature}")
    plt.show()


if __name__ == "__main__":
    plot_dG_feature("pairing", "mean")
    plot_dG_feature("pairing", "median")
    plot_dG_feature("folding", "mean")
    plot_dG_feature("folding", "median")
    plot_heatmap()
    plot_cumulative_cell_distribution()
