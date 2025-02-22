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
    # error bars
    eh = np.sqrt(ph * (1-ph) / r)
    ep = np.sqrt(pp * (1-pp) / r)
    fig = plt.figure()
    plt.plot(ph, linewidth=2, color='tab:orange', label="Observed Success")
    plt.plot(pp, linewidth=2, color='tab:blue', label="Predicted Success")
    plt.fill_between(r-1, ph-eh, ph+eh, color='tab:orange', alpha=0.15)
    #plt.fill_between(r-1, pp-ep, pp+ep, color='tab:blue', alpha=0.33)
    plt.plot([0, limit], [0.8, 0.8], '--', color='k', linewidth=1)
    plt.plot([0, limit], [0.75, 0.75], '--', color='k', linewidth=1)
    plt.ylim(0.65, 0.9)
    plt.xlim(0, limit)
    plt.xlabel("Rank", fontsize="x-large")
    plt.ylabel("Average Cumulative Success", fontsize="x-large")
    plt.legend(fontsize="x-large")
    return fig


def calibration_curve(results, bucket_sizes=[1000]):
    """Plot of predicted success rate vs actual success rate."""
    fig = plt.figure()
    results = results[["proba", "hit"]].sort_values("proba")

    for bucket_size in bucket_sizes:
        tmp = results.rolling(bucket_size).mean()
        h = tmp.hit.values
        plt.plot(
            tmp.proba.values, h, label="Rolling Average(%d)" % bucket_size
        )

    p = np.linspace(0, 1)
    plt.plot(p, p, 'k--', label="Target")
    # TODO(ryan): decide if/how we want to support multiple bucket sizes
    error = 1.96 * np.sqrt(p * (1-p) / bucket_sizes[0])
    plt.fill_between(p, p-error, p+error, color='k', alpha=0.15)
    plt.xlabel("Predicted Probability", fontsize="x-large")
    plt.ylabel("Observed Proportion", fontsize="x-large")
    plt.legend(fontsize="large")
    plt.xlim(0.5, 0.9)
    plt.ylim(0.5, 0.9)
    return fig
