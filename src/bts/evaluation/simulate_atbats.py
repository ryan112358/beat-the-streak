import socket
import argparse
from bts.evaluation import metrics
import pandas as pd
import os
import pickle

def simulate(model, atbats,  pitches = None, test_year = 2016):
    """
    :param model: a Model for prediction
    :param data: a data set
    :param test_year: season to do simulation for
    """
    p_train = pitches[pitches.year < test_year] if pitches is not None else None
    if pitches is not None:
        p_test = pitches[pitches.year == test_year]
        p_groups = p_test.groupby('game_date')
   
    train = atbats[atbats.year < test_year]
    test = atbats[atbats.year == test_year]
    model.train(train, p_train)

    results = []
 
    for date, group in test.groupby('game_date', observed=True):
        print(date, group.shape[0])
        tmp = group[['game_date', 'batter', 'hit']].copy()
        #tmp['proba'] = model.predict(group)[tmp.batter].values
        tmp['proba'] = model.predict(group).values # may contain same batter twice
        results.append(tmp)
        p_group = p_groups.get_group(date) if pitches is not None else None
        model.update(group, p_group)

    return pd.concat(results)
