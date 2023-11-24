import pandas as pd
import os

ROOT = os.environ['BTSDATA']

def load_data():
    """
    Load (player, game) data from the processed pickle file
    """
    return pd.read_parquet(f'{ROOT}/data.parquet.gzip')

def load_atbats():
    """
    Load atbat data from the processed pickle file
    """
    return pd.read_parquet(f'{ROOT}/atbats.parquet.gzip')

def load_pitches():
    """
    Load pitch data from the processed pickle file
    """
    return pd.read_parquet(f'{ROOT}/pitches.parquet.gzip')

def load_weather():
    """
    Load weather data.
    """
    return pd.read_csv(f'{ROOT}/weather/weather.csv')

