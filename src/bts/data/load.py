import pandas as pd
import os
import functools
import glob

ROOT = os.environ["BTSDATA"]


def read_parquet(name, columns=None):
    return pd.read_parquet(f"{ROOT}/{name}.parquet.gzip", columns=columns)


load_data = functools.partial(read_parquet, "data")
load_atbats = functools.partial(read_parquet, "atbats")
load_pitches = functools.partial(read_parquet, "pitches")
load_retrosheet = functools.partial(read_parquet, "retrosheet")
load_batx = functools.partial(read_parquet, "batx")

def load_player_lookup():
    return read_parquet("player_lookup")[0]

def load_engineered_features():
    paths = glob.glob(os.path.join(ROOT, 'engineered', '*'))
    result = {}
    for p in paths:
        key = os.path.basename(p)[:-13] #.strip('.parquet.gzip')
        result[key] = pd.read_parquet(p)
    return result
