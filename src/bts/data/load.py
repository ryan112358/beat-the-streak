import pandas as pd
import os
import functools

ROOT = os.environ['BTSDATA']

def read_parquet(name):
    return pd.read_parquet(f'{ROOT}/{name}.parquet.gzip')

load_data = functools.partial(read_parquet, 'data')
load_atbats = functools.partial(read_parquet, 'atbats')
load_pitches = functools.partial(read_parquet, 'pitches')

def load_engineered_features():
    feature_types = ['batter', 'pitcher', 'ballpark', 'batter_team']
    features = []
    for join_key in feature_types:
        name = '%s_features' % join_key
        features.append(([join_key, 'game_date'], read_parquet(name)))
    return features

