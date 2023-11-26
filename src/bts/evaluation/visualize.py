import numpy as np
import matplotlib.pyplot as plt

""" visualizations for the output of a model simulation on a year of data """


def success_distribution(results):
    """Plot the distribution of the most likely hit probability each day."""
    maxs = results.groupby("game_date").proba.max().values
    fig = plt.figure()
    plt.hist(maxs)
    plt.xlabel("Predicted Probability", fontsize="x-large")
    plt.ylabel("Number of Occurrences", fontsize="x-large")
    return fig


def success_curve(results, limit=1000):
    """Plot of cumulative success rate for most likely batters each day."""

    df = results.sort_values("proba", ascending=False)

    h = df.hit.astype(float).cumsum().values
    p = df.proba.cumsum().values
    r = np.arange(1, h.size + 1)
    ph = h / r
    pp = p / r
    fig = plt.figure()
    plt.plot(ph[:limit], linewidth=2, label="Observed Success")
    plt.plot(pp[:limit], linewidth=2, label="Predicted Success")
    plt.plot([0, limit], [0.8, 0.8], linewidth=2, label="80% Success")
    plt.plot([0, limit], [0.75, 0.75], linewidth=2, label="75% Success")
    plt.ylim(0, 1)
    plt.xlabel("Rank", fontsize="x-large")
    plt.ylabel("Average Cumulative Success", fontsize="x-large")
    plt.legend(fontsize="x-large")
    return fig


def calibration_curve(results, bucket_sizes=[250]):
    """Plot of predicted success rate vs actual success rate."""
    fig = plt.figure()
    results = results[["proba", "hit"]].sort_values("proba")

    for bucket_size in bucket_sizes:
        tmp = results.rolling(bucket_size).mean()
        plt.plot(
            tmp.proba.values, tmp.hit.values, label="Rolling Average(%d)" % bucket_size
        )

    plt.plot(np.linspace(0, 1), np.linspace(0, 1), label="Target")
    plt.xlabel("Predicted Probability", fontsize="x-large")
    plt.ylabel("Observed Proportion", fontsize="x-large")
    plt.legend(fontsize="large")
    return fig
