import pandas as pd
import numpy as np
from scipy.stats import gmean

class Model:
    def __init__(self):
        pass

    def train(self, data, atbats=None, pitches=None):
        """
        Train the model using the data

        :param data: a pandas data frame containing data for each (player, game)
        :param atbats: a pandas data frame of all associated at bats
        :param pitches: a pandas data frame of all associated pitches
        """
        pass

    def update(self, data, atbats=None, pitches=None):
        """
        Update the model using the new data

        :param data: a pandas data frame containing data for each (player, game)
        :param atbats: a pandas data frame of all associated at bats
        :param pitches: a pandas data frame of all associated pitches
        """
        pass
    
    def predict(self, data):
        """
        Predict the probability of a hit for each player in the given data

        :param data: a pandas data frame containing (player, game) data
        :return: a pandas series consisting of hit probability estimates for each batter in data
        """
        pass
    
    def summary(self):
        """
        Return a summary of this model (e.g., model weights)
        """
        pass

    def save(self, path):
        """
        Save the model to the given path
        :param: path to location of saved file
        """
        pass

    def load(self, path):
        """
        Load the model from the given path
        :param: path to location of file to load
        """
        pass

    def __str__(self):
        return self.__class__.__name__

class Baseline(Model):
    """
    The baseline model approximates the probability of a hit by taking the proportion 
    of hits on all historical data for the batter.  All other information is ignored.
    """
    def __init__(self, base = 0.62, scale=40):
        """ 
        :param base: prior probability of getting a hit
        :param scale: confidence on the estimate of prior probability
        """
        self.base = base
        self.scale = scale
    
    def train(self, data, atbats=None, pitches=None):
        totals = data.batter.value_counts()
        hits = data[data.hit].batter.value_counts()
        hits = hits[totals.index].fillna(0)
        self.totals = totals
        self.hits = hits

    def update(self, data, atbats=None, pitches=None):
        totals = data.batter.value_counts()
        hits = data[data.hit].batter.value_counts()
        hits = hits[totals.index].fillna(0)
        self.totals = self.totals.add(totals, fill_value=0)
        self.hits = self.hits.add(hits, fill_value=0)
    
    def predict(self, df):
        hits = self.hits + self.base*self.scale
        totals = self.totals + self.scale
        means = hits.div(totals, fill_value=0)
        pred = means[df.batter].fillna(self.base).values
        return pd.Series(data=pred, index=df.batter.values)

    def __str__(self):
        base = self.base
        scale = self.scale
        name = self.__class__.__name__
        return name + '_' + str(base) + '_' + str(scale)

    def __repr__(self):
        return self.__str__()

class Constant(Model):
    def __init__(self, proba=0.62):
        self.proba = proba

    def predict(self, df):
        pred = np.ones(df.shape[0]) * self.proba
        return pd.Series(data=pred, index=df.batter.values)

    def __str__(self):
        base = self.proba
        name = self.__class__.__name__
        return name + '_' + str(base)

class Ensemble(Model):
    def __init__(self, submodels, aggregation = 'mean', weights = None):
        """
        An ensemble of models, where probability predictions from submodels are combined
        with the given aggregation function
        :param submodels: a list of submodels (of type Model)
        :param aggregation: a string indicating the aggregation function (mean, median, min, geom)
        :param weights: an optional argument for when aggregation = "mean" to weight each submodel
        """
        self.submodels = submodels
        self.aggregation = aggregation
        self.weights = weights
        assert aggregation in ['mean', 'median', 'min', 'geom'], 'invalid aggregation value'
    
    def train(self, data, atbats, pitches):
        for model in self.submodels:
            model.train(data, atbats, pitches)
    
    def update(self, data, atbats, pitches):
        for model in self.submodels:
            model.update(data, atbats, pitches)
    
    def predict(self, data):
        N = len(self.submodels)
        P = np.zeros((N, data.shape[0]))
        bats = data.batter.values
        for i in range(N):
            P[i] = self.submodels[i].predict(data)[bats]
        if self.aggregation == 'mean':
            ans = np.mean(P, axis=0)
            if self.weights:
                ans = np.dot(P.T, self.weights)
        elif self.aggregation == 'median':
            ans = np.median(P, axis=0)
        elif self.aggregation == 'min':
            ans = np.min(P, axis=0)
        elif self.aggregation == 'geom':
            ans = gmean(P, axis=0)
        return pd.Series(data=ans, index=bats)
