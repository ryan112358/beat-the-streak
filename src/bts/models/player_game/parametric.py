import jax
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
from bts.data.load import load_engineered_features
from bts.models import neural_network
from jax import config
import jax.numpy as jnp

config.update('jax_enable_x64', True)

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
        cols = ['order', 'ballpark', 'batter', 'pitcher', \
                'home', 'batter_team', 'pitcher_team', \
                'stand', 'p_throws']
#                'stand_p_throws', 'ballpark_stand', 'batter_home', 'batter_p_throws']
        df = data.copy()
        self._engineer(df)
        self.transformer = OneHotEncoder(sparse=True, handle_unknown='ignore')
        self.transformer.fit(df[cols].values)

    def _onehot(self, data):
        cols = ['order', 'ballpark', 'batter', 'pitcher', \
                'home', 'batter_team', 'pitcher_team', \
                'stand', 'p_throws']#  \
#                'stand_p_throws', 'ballpark_stand', 'batter_home', 'batter_p_throws']

        df = data.copy()
        self._engineer(df)
        return self.transformer.transform(df[cols].values)        

    def _engineer(self, data):
        utility.combine(data, 'stand', 'p_throws')
        utility.combine(data, 'ballpark', 'stand')
        utility.combine(data, 'batter', 'home')
        utility.combine(data, 'batter', 'p_throws')
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

    def train(self, data, atbats=None, pitches=None):
       
        self._fit_transformer(data) 
        self.X = self._onehot(data)
        self.y = data.hit.values


        dates = pd.to_datetime(data.game_date)
        self.dates = dates
        deltas = (dates.max() - dates).dt.days
        weights = self.alpha**deltas.values

        self.model.fit(self.X, self.y, sample_weight=weights)
    
    def update(self, data, atbats=None, pitches=None):
        #print(data.iloc[0].date)
        X = self._onehot(data)
        y = data.hit.values
        self.X = scipy.sparse.vstack([self.X, X])
        self.y = np.append(self.y, y)

        dates = pd.concat([self.dates, data.game_date])
        self.dates = dates
        deltas = (dates.max() - dates).dt.days
        weights = self.alpha**deltas.values

        self.model.fit(self.X, self.y, sample_weight=weights)
    
    def predict(self, data):
        X = self._onehot(data)
        pred = self.model.predict_proba(X)[:,1]
        return pd.Series(data=pred, index=data.batter.values)

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


class Singlearity(model.Model):
    def __init__(self, name='Singlearity'):
        self.model = neural_network.FullyConnectedNeuralNetwork()
        self.name = name
        self.base_features = ['home', 'game_year', 'order', 'stand', 'p_throws']
        self.engineered = load_engineered_features()
        self.Xmean = None
        self.Xstd = None

    def _feature_join(self, data):
        merged = data
        for join_key, table in self.engineered:
            merged = merged.merge(table, on=join_key, how='left')
        return merged

    def fit(self, steps):
        for i, rng in enumerate(jax.random.split(self.rng, steps)):
            grads, loss, mean_hits, accuracy, pred = neural_network.apply_model(self.state, self.X, self.y, {'dropout': rng})
            self.state = neural_network.update_model(self.state, grads)
            if i % 50 == 0:
                print(loss, mean_hits, accuracy, jnp.isnan(pred).any())
        self.rng = rng

    def get_Xy(self, data):       
        merged = self._feature_join(data)
        basic = pd.get_dummies(merged[self.base_features])
        engineered = merged.filter(regex='1095|365|30|60')
        df = pd.concat([basic, engineered], axis=1)
        idx = merged.game_year >= 2003
        # from IPython import embed; embed()
        X = df[idx].values.astype(float)
        y = merged[idx].hit.values.astype(float)
        if self.Xstd is None:
            Xmean = X.mean(axis=0)
            Xstd = X.std(axis=0)
            self.Xindex = Xstd > 0
            self.Xstd = Xstd[self.Xindex]
            self.Xmean = Xmean[self.Xindex]
        X = (X[:,self.Xindex] - self.Xmean) / self.Xstd
        return X, y

    def summary(self):
        return None

    def train(self, data, atbats=None, pitches=None):
        self.X, self.y = self.get_Xy(data)
        self.rng = jax.random.PRNGKey(0)
        self.state = neural_network.create_train_state(self.model, self.rng, self.X)
        self.fit(8000)
        
 
    def update(self, data, atbats=None, pitches=None):
        X, y = self.get_Xy(data)
        self.X = np.vstack([self.X, X])
        self.y = np.append(self.y, y)
        self.fit(250)
    
    def predict(self, data):
        X, _ = self.get_Xy(data)
        params = {'params': self.state.params}
        pred = self.model.apply(params, X, train=False).flatten()
        return pd.Series(data=pred, index=data.batter.values)

    def __str__(self):
        return self.name
