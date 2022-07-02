import pandas as pd
from pandas.api.types import CategoricalDtype
from itertools import product
import datetime as dt

def combine(df, *features):
    """
    Create a crossed-column feature two or more categorical features
    The contents of this new feature is the tuple of the existing features
    and is a useful for feature engineering before applying some ML model
    """
    col = '_'.join(features)
    data = list(zip(*[df[a] for a in features]))
    df[col] = data 
    #cols = [df[a].astype('category') for a in features]
    
def dayoff(df):
    """
    Derives a new feature for the data, dayoff, which is true if the batter 
    didn't play on the previous day, and false otherwise
    """
    df['date'] = pd.to_datetime(df.date)
    tmp = df[['batter', 'date']].copy()
    tmp['prev'] = df.date - dt.timedelta(days=1)
    merged = tmp.merge(tmp, left_on=['batter','date'], right_on=['batter', 'prev'])
    merged = merged[['batter', 'date_y']].rename(columns = { 'date_y' : 'date' })
    merged['dayoff'] = False
    tmp = tmp.merge(merged, how='left', on=['batter', 'date'])
    df['dayoff'] = tmp.dayoff.fillna(True).values

def data_to_atbats(data):
    converter = { 'bstand' : 'b_stand', 'pthrows' : 'p_throws' }
    ans = data.rename(columns = converter)
    ans['starting_pitcher'] = ans['pitcher']
    return ans
