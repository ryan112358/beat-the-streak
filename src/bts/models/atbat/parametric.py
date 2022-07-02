from bts.models.player_game import model, utility
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import lightgbm
import xgboost
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy
from scipy import sparse
import pickle

def onehot(data, mincount=50):
    cats = []
    for col in data.columns:
        counts = data[col].value_counts()
        cats.append( list( (counts > 50).index ) )
    return OneHotEncoder(categories=cats, handle_unknown='ignore')

class SKLearn(model.Model):
    def __init__(self, base_model, alpha=0.9995, name='SKLearn'):
        self.model = base_model
        self.alpha = alpha
        self.name = name

    def _fit_transformer(self, data):
        cols = ['batter', 'pitcher', 'stand', 'p_throws', 'ballpark', 'on_3b', 'on_2b', 'on_1b',  \
                'outs_when_up', 'inning', \
                'home', 'umphome', 'temp', 'winddir', 'windspeed', 'precip', 'sky']
        df = data.copy()
        #self._engineer(df)
        self.transformer = OneHotEncoder(sparse=True, handle_unknown='ignore')
        self.transformer.fit(df[cols].values)

    def _onehot(self, data):
        cols = ['batter', 'pitcher', 'stand', 'p_throws', 'ballpark', 'on_3b', 'on_2b', 'on_1b',  \
                'outs_when_up', 'inning', \
                'home', 'umphome', 'temp', 'winddir', 'windspeed', 'precip', 'sky']
 
        df = data.copy()
        #self._engineer(df)
        return self.transformer.transform(df[cols].values)        

    def _engineer(self, data):
        utility.combine(data, 'stand', 'p_throws')
        utility.combine(data, 'ballpark', 'stand')
        #utility.combine(data, 'batter', 'home')
        #utility.combine(data, 'batter', 'p_throws')
        #utility.combine(data, 'batter', 'year') # doesn't work
        return data
    
    def _weights(self, feature=None, sort=True):
        ans = pd.Series(index=self.features, data=self.model.coef_.flatten())
        if feature:
            ans = ans.filter(like=feature)
        if sort:
            ans = ans.sort_values(ascending=False)
        return ans
    
    def summary(self):
        return self._weights()

    def train(self, atbats, pitches=None):
       
        self._fit_transformer(atbats) 
        self.X = self._onehot(atbats).astype(np.float32)
        self.y = atbats.hit.values.astype(np.int64)


        dates = pd.to_datetime(atbats.game_date)
        self.dates = dates
        deltas = (dates.max() - dates).dt.days
        weights = self.alpha**deltas.values

        if self.alpha == 1:
            self.model.fit(self.X, self.y)
        else:
            self.model.fit(self.X, self.y, sample_weight=weights)
    
    def update(self, atbats, pitches=None):
        #print(data.iloc[0].date)
        X = self._onehot(atbats).astype(np.float32)
        y = atbats.hit.values.astype(np.int64)
        self.X = scipy.sparse.vstack([self.X, X])
        self.y = np.append(self.y, y)

        dates = self.dates.append(pd.to_datetime(atbats.game_date))
        self.dates = dates
        deltas = (dates.max() - dates).dt.days
        weights = self.alpha**deltas.values

        if self.alpha == 1:
            self.model.fit(self.X, self.y)
        else:
            self.model.fit(self.X, self.y, sample_weight=weights)
    
    def predict(self, atbats):
        X = self._onehot(atbats).astype(np.float32)
        pred = self.model.predict_proba(X)[:,1]
        return pd.Series(data=pred, index=atbats.batter.values)

    def __str__(self):
        return self.name

def Logistic(alpha=0.9995, reg=4.0, penalty='l2'):
    # l1+saga: 1m25s, l1+liblinear: 3m21s, l2+lbfgs: 45s
    if penalty == 'l1':
        model = LogisticRegression('l1',C=1.0/reg,solver='saga',warm_start=True,n_jobs=-1)
    else:
        model = LogisticRegression('l2',C=1.0/reg,solver='lbfgs',warm_start=True,n_jobs=-1,max_iter=1000)
    name = 'Logistic_' + str(alpha) + '_' + str(reg) + '_' + str(penalty)
    return SKLearn(model, alpha, name)

def LightGBM(alpha=0.9995):
    model = lightgbm.LGBMClassifier()
    name = 'LightGBM_' + str(alpha)
    return SKLearn(model, alpha, name)

def XGBoost(alpha=0.9995):
    model = xgboost.XGBClassifier()
    name = 'XGBoost_' + str(alpha)
    return SKLearn(model, alpha, name)
