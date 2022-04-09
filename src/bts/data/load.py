import pandas as pd
import os

ROOT = os.environ['BTSDATA']

def load_data():
    """
    Load (player, game) data from the processed pickle file
    """
    return pd.read_pickle('%s/data.pkl' % ROOT)

def load_atbats():
    """
    Load atbat data from the processed pickle file
    """
    return pd.read_pickle('%s/atbats.pkl' % ROOT)

def load_pitches():
    """
    Load pitch data from the processed pickle file
    """
    return pd.read_pickle('%s/pitches.pkl' % ROOT)

