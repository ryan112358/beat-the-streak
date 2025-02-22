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
from IPython import embed

config.update("jax_enable_x64", True)

"""
Note to self:  this is still in a work-in-progress state.  With the new feature format,
I have not yet gotten model convergence.  But also haven't tried much yet.  Should be
doable with a bit more work.
"""


class BTSNet(model.Model):
    def __init__(
        self,
        learning_rate=1e-6,
        regularization=0,
        reg2=1e4,
        hidden_dims=512,
        layers=1,
        log=False,
    ):
        self.model = neural_network.FullyConnectedNeuralNetwork(
            hidden_dims=hidden_dims, layers=layers
        )
        self.layers = layers
        self.hidden_dims = hidden_dims
        self.regularization = regularization
        self.base_features = ["home", "order", "stand", "p_throws"]
        self.engineered = load_engineered_features()
        self.learning_rate = learning_rate
        self.reg2 = reg2
        self.log = log
        self.rng = jax.random.PRNGKey(0)
        self.state = None

    def get_Xy(self, data):
        y = data["ehit_at_least_one"].values.astype(float)
        imputed = np.zeros_like(y)
        X = {
            "base": (
                pd.get_dummies(data[self.base_features]).values.astype(float),
                imputed,
            )
        }
        for key, lookup in self.engineered.items():
            index = pd.Index(data[lookup.index.names])

            F = lookup.reindex(index)
            if "imputed" in F.columns:
                imputed = F["imputed"].values.astype(float)
            else:
                imputed = F.notnull().all(axis=1).values.astype(float)

            X[key] = (
                lookup.reindex(index)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .values.astype(float),
                imputed,
            )
        return X, y

    def fit(self, data, steps):
        X, y = self.get_Xy(data)
        if self.state is None:
            state = neural_network.create_train_state(
                self.model, self.rng, X, self.learning_rate
            )
        else:
            state = self.state

        total_loss = 0
        max_prob = 0
        for i, rng in enumerate(jax.random.split(self.rng, steps)):
            grads, loss, pred = neural_network.apply_model(
                state,
                X,
                y,
                rng={"dropout": rng},
                regularization=self.regularization,
                reg2=self.reg2,
                previous_state=self.state,
            )
            if i == 0 and self.log:
                print("Loss Before", loss)
                print("Count Before[>0.75]", np.sum(pred > 0.75))
                print("Accuracy Before[>0.75]", np.mean(y[pred > 0.75]))
            state = neural_network.update_model(state, grads)
        if self.log and self.state is not None:
            print("Loss After", loss)
            print("Count After[>0.75]", np.sum(pred > 0.75))
            print("Accuracy After[>0.75]", np.mean(y[pred > 0.75]))
            print("Max Prob", pred.max())
            if pred.min() < 0 or pred.max() > 1:
                embed()
            print()
            #print(
            #    "Model Diff",
            #    jnp.sqrt(
            #        neural_network.model_squared_difference(
            #            state.params, self.state.params
            #        )
            #    ),
            #)
            #print("Model Norm", neural_network.optax.global_norm(state.params))
            # print("X min/max", X.min(), X.max())  # monitor, explains instability
        self.rng = rng
        self.state = state

    def summary(self):
        return None

    def train(self, data, atbats=None, pitches=None):
        for _, group in data.groupby("game_date"):
            self.update(group)

    def update(self, data, atbats=None, pitches=None):
        self.fit(data, 25)

    def predict(self, data):
        X, _, = self.get_Xy(data)
        params = {"params": self.state.params}
        phit = self.model.apply(params, X, train=False).flatten()
        return pd.Series(data=phit, index=data.batter.values)

    def __str__(self):
        return f"BTSNet_{self.regularization}_{self.reg2}_{self.learning_rate}_{self.layers}_{self.hidden_dims}"
