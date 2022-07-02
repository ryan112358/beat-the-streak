#from bts.models.atbat.parametric import SKLearn
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from bts.models.atbat import model
import numpy as np
import torch
from torch.optim import SGD
from torch import nn
import torch.nn.functional as F
import pandas as pd

gpu = torch.device("cuda:0")

class BatterPitcherModel(nn.Module):
    def __init__(self, layers=0, hidden_size=50):
        super(BatterPitcherModel, self).__init__()
        self.batter_embedding = nn.Embedding(3104, 10)
        self.pitcher_embedding = nn.Embedding(2488, 10)
        self.first = nn.Linear(20, hidden_size)
        self.inner = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers)])
        self.last = nn.Linear(hidden_size, 2)

    def forward(self, X):
        B = self.batter_embedding(X[:,0])
        P = self.pitcher_embedding(X[:,1])
        x = torch.cat([B,P], 1)
        x = F.relu(self.first(x))
        for lin in self.inner:
            x = F.relu(lin(x))
        x = F.relu(self.last(x))
        return x

class FullModel(nn.Module):
    def __init__(self, layers=0, hidden_size=50, num_categories=[]):
        super(FullModel, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, 4) for n in num_categories])
        print(num_categories)
        self.first = nn.Linear(4*len(num_categories), hidden_size)
        self.inner = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers)])
        self.last = nn.Linear(hidden_size, 2)

    def forward(self, Xcat, Xnum):
        xs = [embed(Xcat[:,i]) for i, embed in enumerate(self.embeddings)]
        x = torch.cat(xs, 1)
        x = F.relu(self.first(x))
        for lin in self.inner:
            x = F.relu(lin(x))
        x = F.relu(self.last(x))
        return x

   

class NeuralNet(model.Model):
    def __init__(self):
        self.cat_cols = ['batter', 'pitcher', 'stand', 'p_throws', 'ballpark', 'umphome', 'winddir', 'precip', 'sky']
        self.num_cols = ['home', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'temp', 'windspeed']

    def train(self, atbats, pitchers=None):
        self.y = torch.tensor(atbats.hit.values.astype(np.int64)).to(gpu)
        num_categories = [atbats[col].astype('category').cat.categories.size for col in self.cat_cols]
        self.model = FullModel(layers=3, hidden_size=50, num_categories=num_categories).to(gpu)
        self.optim = SGD(self.model.parameters(), lr=0.05)

        self.Xcat = torch.tensor(atbats[self.cat_cols].apply(lambda c: c.cat.codes).values.astype(np.int64)).to(gpu)
        self.Xnum = torch.tensor(atbats[self.num_cols].values.astype(np.int64)).to(gpu)
 
        self.model.train() 
        for _ in range(250):
            out = self.model(self.Xcat, self.Xnum)
            loss = F.cross_entropy(out, self.y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step() 

    def update(self, atbats, pitchers):
        y = torch.tensor(atbats.hit.values.astype(np.int64)).to(gpu)
        Xcat = torch.tensor(atbats[self.cat_cols].apply(lambda c: c.cat.codes).values.astype(np.int64)).to(gpu)
        Xnum = torch.tensor(atbats[self.num_cols].values.astype(np.int64)).to(gpu)
        self.y = torch.cat([self.y, y])
        self.Xcat = torch.cat([self.Xcat, Xcat], 0)
        self.Xnum = torch.cat([self.Xnum, Xnum], 0)

        self.model.train() 
        for _ in range(25):
            out = self.model(self.Xcat, self.Xnum)
            loss = F.cross_entropy(out, self.y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step() 

    def predict(self, atbats):
        Xcat = torch.tensor(atbats[self.cat_cols].apply(lambda c: c.cat.codes).values.astype(np.int64)).to(gpu)
        Xnum = torch.tensor(atbats[self.num_cols].values.astype(np.int64)).to(gpu)

        self.model.eval()
        pred = F.softmax(self.model(Xcat, Xnum), dim=1).cpu()[:,1].detach().numpy()
        #print(pred)
        return pd.Series(data=pred, index=atbats.batter.values)

class TabNet(model.Model):
    def __init__(self):
        torch.cuda.empty_cache()
        self.model = TabNetClassifier(
            cat_idxs=[0,1], 
            cat_dims=[3104,2488], 
            cat_emb_dim=10)
        self.name = 'TabNet'

    def train(self, atbats, pitches=None):
       
        self.X = atbats[['batter', 'pitcher']].apply(lambda c: c.cat.codes).values
        self.y = atbats.hit.values.astype(np.int64)
        
#        from IPython import embed; embed()

        self.model.fit(self.X, self.y, batch_size=2**15)
    
    def update(self, atbats, pitches=None):
        X = atbats[['batter', 'pitcher']].apply(lambda c: c.cat.codes).values
        y = atbats.hit.values.astype(np.int64)
        self.X = np.vstack([self.X, X])
        self.y = np.append(self.y, y)

        self.model.fit(self.X, self.y, batch_size=2**15)
    
    def predict(self, atbats):
        X = self.atbats[['batter', 'pitcher']].apply(lambda c: c.cat.codes)
        pred = self.model.predict_proba(X)[:,1]
        return pd.Series(data=pred, index=atbats.batter.values)

    def __str__(self):
        return self.name
