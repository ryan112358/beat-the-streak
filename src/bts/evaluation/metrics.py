"""Metrics for evaluating the quality of predictions for Beat the Streak."""

import numpy as np
import functools


def likelihood(results):
    """ 
    Geometric mean likelihood of estimates

    :param probas: a list of numpy arrays (one per day) that contain hit probability estimates 
    :param hits: a list of numpy arrays (one per day) that contain binary hit indicator variables
    :return: the likelihood of the estimates
    """
    probas = results["proba"].values
    hits = results["hit"].values.astype(float)
    ans = hits * np.log(probas) + (1.0 - hits) * np.log(1.0 - probas)
    return np.exp(np.mean(ans))


def accuracy(results, top=5):
    """
    Percentage of correct predictions for top k batters every day

    :param results: a pandas data frame containing columns for date, batter, hit, proba
    :param top: number of batters to use in evaluating accuracy (will take the most likely ones) 
    :return: the accuracy of the predictions
    """
    topk = results.groupby("game_date").proba.nlargest(top)
    idx = topk.index.get_level_values(1)
    return results.loc[idx].hit.mean()


def conditional_success(results, threshold=0.78):
    return results[results.proba >= threshold].hit.mean()


def conditional_count(results, threshold=0.78):
    return results[results.proba >= threshold].shape[0]


def best_accuracy(results, top=300):
    return results.nlargest(top, "proba").hit.mean()


def brier(results):
    groups = (results.proba * 100).round().astype(int)
    N = float(results.shape[0])

    o = np.mean(results.hit)
    rel, res, unc = 0, 0, o * (1 - o)

    for _, g in results.groupby(groups):
        nk = g.shape[0]
        fk = g.proba.mean()
        ok = g.hit.mean()
        rel += nk / N * (fk - ok) ** 2
        res += nk / N * (ok - o) ** 2

    return rel, res, unc


def reliability(results):
    return brier(results)[0]


def resolution(results):
    return brier(results)[1]


def mse(results):
    p = results.proba
    o = results.hit
    return ((p - o) ** 2).mean()


def custom_score(probas, hits, top=10):
    """
    score the predictions using custom scoring metric

    :param probas: a list of numpy arrays of hit probabilities (one for each day)
    :param hits: a list of numpy arrays of hit outcomes (one for each day)
    :param top: number of batters to use in evaluating accuracy (will take the most likely ones) 
    :return: the score (higher is better)
    """
    score = 0
    for p, h in zip(probas, hits):
        if len(h) >= top:
            idx = np.argsort(p)
            score += np.arange(top) * h[idx][-top:]
    return score


ALL_METRICS = {
    "Top 2/Day Accuracy": functools.partial(accuracy, top=2),
    "Top 5/Day Accuracy": accuracy,
    "Top 300 Accuracy": best_accuracy,
    "Accuracy(proba>0.78)": conditional_success,
    "Count(proba>0.78)": conditional_count,
    "Likelihood": likelihood,
    "MSE": mse,
}
