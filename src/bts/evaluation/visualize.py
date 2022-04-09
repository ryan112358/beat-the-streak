import numpy as np
import matplotlib.pyplot as plt

""" visualizations for the output of a model simulation on a year of data """

def success_distribution(results):
    """
    plot the distribution of the most likely hit probability each day

    """
    maxs = results.groupby('game_date').proba.max().values
    fig = plt.figure()
    plt.hist(maxs)
    plt.xlabel('Predicted Probability', fontsize='x-large')
    plt.ylabel('Number of Occurrences', fontsize='x-large')
    return fig 

def success_curve(results, limit = 1000):
    """
    :return: plot of cumulative success rate for most likely hits
    """

    df = results.sort_values('proba', ascending=False)

    h = df.hit.astype(float).cumsum().values
    p = df.proba.cumsum().values
    r = np.arange(1, h.size+1)
    ph = h / r
    pp = p / r
    fig = plt.figure()
    plt.plot(ph[:limit], linewidth=2, label='Observed Success')
    plt.plot(pp[:limit], linewidth=2, label='Predicted Success')
    plt.plot([0, limit], [0.8, 0.8], linewidth=2, label='80% Success')
    plt.plot([0, limit], [0.75, 0.75], linewidth=2, label='75% Success')
    plt.ylim(0,1)
    plt.xlabel('Rank', fontsize='x-large')
    plt.ylabel('Average Cumulative Success', fontsize='x-large')
    plt.legend(fontsize='x-large')
    return fig

def calibration_curve(results, bucket_size=100):
    """
    :param probas: 
    :param hit: 
    :return: plot of predicted success rate vs actual success rate
    """
    probas = results.proba.values
    hits = results.hit.values

    idx = np.argsort(probas)
    probas = probas[idx]
    hits = hits[idx].astype(float)
    split = hits.size / bucket_size
    x = [np.mean(x) for x in np.array_split(probas, split)]
    y = [np.mean(y) for y in np.array_split(hits, split)]
    #print x[-4:],  y[-4:]
    fig = plt.figure()
    plt.plot(x, y, '.')
    plt.plot(np.linspace(0,1), np.linspace(0,1))
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Proportion')
    return fig

