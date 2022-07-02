from bts.models.player_game import model
import numpy as np
import pandas as pd

""" This file contains models that attempt to exploit the sequential nature of the problem """

class ExpSmoothing(model.Model):
    """
    The exponential smoothing model is similar to the baseline in that it only takes into account 
    the information about the batter (and not the pitcher or other contextual information), 
    but it does not weight all data equally likely... instead more recent observations carry more
    weight in the estimation of the hit probability.  The amount of influence that new data points
    have is controlled by alpha.
    """
    def __init__(self, base=0.62, alpha=0.005):
        """
        :param base: the default probability of a hit assigned to new batters
        :param alpha: the exponential smoothing parameter
        """
        self.base = base
        self.alpha = alpha
        self.probas = pd.Series()

    def train(self, data, atbats=None, pitches=None):
        for date, group in data.groupby('game_date'):
            self.update(group)
    
    def update(self, data, atbats=None, pitches=None):
        up = data.set_index('batter').hit.astype(float)
        new_index = self.probas.index.union(up.index.astype(str))
        self.probas = self.probas[new_index].fillna(self.base)
        self.probas[up.index] *= (1-self.alpha)
        self.probas[up.index] += self.alpha*up
    
    def predict(self, data):
        bats = data.batter
        pred = self.probas[bats].fillna(self.base).values
        return pd.Series(data=pred, index=bats)
    
    def __str__(self):
        name = self.__class__.__name__
        base = self.base
        alpha = self.alpha
        return name + '_' + str(base) + '_' + str(alpha)

class EloSystem(model.Model):
    """
    The EloSystem model uses both the batter and the pitcher to determine the probabiliity of a hit.
    All batters and pitchers have an elo rating (with appropriate default value for new players) 
    and from these values we assume there is a simple formula for the probability that the batter
    will get a hit in the game where the pitcher is starting: 
        P(hit) = 1 / (1 + exp(pitcher_elo - batter_elo))
    as new data comes in the elo ratings for the players involved are updated in a sequential fashion
    """
    def __init__(self, bfill=0.0, pfill=-0.6, k_factor=0.05):
        self.k_factor = k_factor
        self.bfill = bfill
        self.pfill = pfill
        self.belo = pd.Series()
        self.pelo = pd.Series()

    def train(self, data, atbats=None, pitches=None):
        for date, group in data.groupby('game_date'):
            self.update(group)
 
    def update(self, data, atbats=None, pitches=None):
        bats = data.batter
        pits = data.pitcher
    
        bindex = self.belo.index.union(bats.unique().astype(str)).unique()
        pindex = self.pelo.index.union(pits.unique().astype(str)).unique()
        self.belo = self.belo.reindex(bindex).fillna(self.bfill)
        self.pelo = self.pelo.reindex(pindex).fillna(self.pfill)

        diff = self.belo[bats].values - self.pelo[pits].values

        ebat = 1.0 / (1.0 + np.exp(-diff))
        sbat = data.hit.values.astype(float)
        bup = pd.Series(data=sbat-ebat, index=bats).groupby('batter').sum()

        df = pd.DataFrame()
        df['pitcher'] = pits.values
        df['dpit'] = ebat - sbat
        pup = df.groupby('pitcher').dpit.sum()

        self.belo = self.belo.add(self.k_factor*bup, fill_value=0)
        self.pelo = self.pelo.add(self.k_factor*pup, fill_value=0)
    
    def predict(self, data):
        bats = data.batter
        pits = data.pitcher
        diff = self.belo.reindex(bats).fillna(self.bfill).values - self.pelo.reindex(pits).fillna(self.pfill).values
        pred = 1.0 / (1.0 + np.exp(-diff))
        return pd.Series(data=pred, index=bats)

    def __str__(self):
        name = self.__class__.__name__
        k_factor = self.k_factor
        bfill = self.bfill
        pfill = self.pfill
        return name + '_' + str(bfill) + '_' + str(pfill) + '_' + str(k_factor)

