"""Derive new features from raw data based on sliding window statistics."""
import numpy as np
import pandas as pd
import os
from IPython import embed
from bts.data.load import *
from glob import glob

def compute_park_factors(atbats):
    # Some test set leakage occurs here, ignoring for now
    # TODO: compute the 3year moving average park factor
    stuff = atbats.copy()
    columns = ['single', 'double', 'triple', 'home_run']
    for e in columns:
        stuff[e] = atbats.events == e
    columns += ['hit']
    batters = stuff.groupby(['home', 'batter_team'])[columns].sum()
    pitchers = stuff.groupby(['home', 'pitcher_team'])[columns].sum()
    games = stuff.groupby(['home', 'batter_team']).game_pk.nunique()
    #return batters, pitchers, games
    numerator = (batters.loc[True] + pitchers.loc[False]).div(games.loc[True], axis=0)
    denominator = (batters.loc[False] + pitchers.loc[True]).div(games.loc[False], axis=0)
    return (numerator / denominator).rename(columns=lambda s: 'pf_%s'%s).fillna(1.0)

def augment_park_factors(atbats):
    pfs = compute_park_factors(atbats)
    pfs.index.name = 'ballpark'
    aug = pfs.loc[atbats.ballpark]
    aug.index = atbats.index
    return pd.concat([atbats, aug], axis=1)

def sliding_mean(df, days=365, thresh=40):
    roll = df.rolling('%dd'%days, on='game_date', closed='left', min_periods=thresh)
    result = roll.mean()
    return result.groupby('game_date').head(n=1)

def sliding_mean_by_partition(df, groupby_keys, columns, days=365, thresh=40):
    groups = df.groupby(groupby_keys, observed=True)[['game_date']+columns]
    apply_fn = lambda df: sliding_mean(df, days, thresh)
    average = apply_fn(df[['game_date']+columns]).fillna(method='bfill')
    per_partition = groups.apply(apply_fn).reset_index(level=0).reset_index(drop=True)
    imputed_vals = per_partition.drop(columns=columns).merge(average, how='left', on='game_date')
    assert imputed_vals.shape[0] == per_partition.shape[0]
    per_partition.update(imputed_vals, overwrite=False)
    rename_fn = lambda s: '_'.join(groupby_keys + [s, str(days)]) if not s in ['game_date']+groupby_keys else s
    return per_partition.rename(columns=rename_fn)

def park_factor_3year(atbats):
    days = 365*3
    events = ['single', 'double', 'triple', 'home_run', 'hit', 'woba']
    home_hits = sliding_mean_by_partition(atbats, ['home_team'], events, days, 40)
    away_hits = sliding_mean_by_partition(atbats, ['away_team'], events, days, 40)
    home_hits = home_hits.rename(columns={'home_team':'ballpark'}).sort_values('game_date')
    away_hits = away_hits.rename(columns={'away_team':'ballpark'}).sort_values('game_date')
    tmp = pd.merge_asof(home_hits, away_hits, on='game_date', by='ballpark')
    for e in events:
        col1 = 'home_team_%s_%d' % (e, days)
        col2 = 'away_team_%s_%d' % (e, days)
        col = 'pf_%s_%d' % (e, days)
        tmp[col] = tmp.pop(col1) / tmp.pop(col2)
    return tmp
        

def batter_3year(atbats):
    events = ['pf_hit_1095', 'woba', 'field_out', 'strikeout', 'single', 'walk', 'double', 'home_run', 'force_out', 'grounded_into_double_play', 'hit_by_pitch', 'field_error', 'sac_fly', 'sac_bunt', 'triple', 'intent_walk', 'double_play']
    return sliding_mean_by_partition(atbats, ['batter'], events, 365*3, 40)

def pitcher_3year(atbats):
    events = ['pf_hit_1095', 'woba', 'field_out', 'strikeout', 'single', 'walk', 'double', 'home_run', 'force_out', 'grounded_into_double_play', 'hit_by_pitch', 'field_error', 'sac_fly', 'sac_bunt', 'triple', 'intent_walk', 'double_play']
    return sliding_mean_by_partition(atbats, ['pitcher'], events, 365*3, 40)

def batter_1year(atbats):
    events = ['pf_hit_1095', 'woba', 'field_out', 'strikeout', 'single', 'walk', 'double', 'home_run', 'force_out', 'grounded_into_double_play', 'hit_by_pitch', 'field_error', 'sac_fly', 'sac_bunt', 'triple', 'intent_walk', 'double_play']
    return sliding_mean_by_partition(atbats, ['batter'], events, 365, 40)

def pitcher_1year(atbats):
    events = ['pf_hit_1095', 'woba', 'field_out', 'strikeout', 'single', 'walk', 'double', 'home_run', 'force_out', 'grounded_into_double_play', 'hit_by_pitch', 'field_error', 'sac_fly', 'sac_bunt', 'triple', 'intent_walk', 'double_play']
    return sliding_mean_by_partition(atbats, ['pitcher'], events, 365, 40)

def batter_recent(atbats):
    events = ['pf_hit_1095', 'single', 'double', 'triple', 'home_run', 'walk', 'strikeout', 'woba']
    return sliding_mean_by_partition(atbats, ['batter'], events, 21, 20)

def pitcher_recent(atbats):
    events = ['pf_hit_1095', 'single', 'double', 'triple', 'home_run', 'walk', 'strikeout', 'woba']
    return sliding_mean_by_partition(atbats, ['pitcher'], events, 21, 20)

def batter_vs_pitcher_3year(atbats):
    events = ['pf_hit_1095', 'single', 'double', 'triple', 'home_run', 'walk', 'strikeout', 'woba']
    return sliding_mean_by_partition(atbats, ['batter', 'pitcher'], events, 365*3, 10)

def merge_features(features, key):
    df = features[0]
    for feature in features[1:]:
        df = pd.merge(df, feature, on=key, how='left')
    return df

def all_batter_features(atbats):
    features = [batter_3year(atbats), batter_1year(atbats), batter_recent(atbats)]
    return merge_features(features, ['batter', 'game_date'])

def all_pitcher_features(atbats):
    features = [pitcher_3year(atbats), pitcher_1year(atbats), pitcher_recent(atbats)]
    return merge_features(features, ['pitcher', 'game_date'])

def all_park_features(atbats):
    features = [park_factor_3year(atbats)]
    return merge_features(features, ['ballpark', 'game_date'])

def all_batter_team_features(atbats):
    # TODO: Move this logic to a helper function
    pa_per_game = atbats.groupby(['batter_team', 'game_pk', 'game_date'], observed=True).hit.count().rename('PA').reset_index().sort_values('game_date')
    pa = sliding_mean_by_partition(pa_per_game, ['batter_team'], ['PA'], 60, 10)
    features = [pa]
    return merge_features(features, ['batter_team', 'game_date'])
    
if __name__ == '__main__':
    atbats = load_atbats()

    dummy = pd.get_dummies(atbats['events'])
    atbats = pd.concat([atbats, dummy], axis=1)
    weights = dict(walk=0.69, hit_by_pitch=0.719, single=0.87, double=1.217, triple=1.529, home_run=1.94)
    atbats['woba'] = sum(weights[e]*atbats[e] for e in weights)
    park_features = all_park_features(atbats)
    atbats = atbats.merge(park_features, how='left', on=['ballpark', 'game_date'])

    batter_features = all_batter_features(atbats)
    pitcher_features = all_pitcher_features(atbats)
    batter_team_features = all_batter_team_features(atbats)

    ROOT = os.environ['BTSDATA']
    park_features.to_parquet(f'{ROOT}/ballpark_features.parquet.gzip', compression='gzip')
    batter_features.to_parquet(f'{ROOT}/batter_features.parquet.gzip', compression='gzip')
    pitcher_features.to_parquet(f'{ROOT}/pitcher_features.parquet.gzip', compression='gzip')
    batter_team_features.to_parquet(f'{ROOT}/batter_team_features.parquet.gzip', compression='gzip')

    #engineered_data = engineered_data[engineered_data.game_date >= '2013-01-01']
    #num_data = engineered_data.filter(regex='1095|365|21|60')
