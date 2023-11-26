"""Simualte a model on historical batter/game data."""
import socket
import matplotlib.pyplot as plt
import argparse
from bts.evaluation import metrics, visualize
import pandas as pd
import os
import pickle
from bts.models.player_game import model, sequential
from bts.data.load import load_data, load_atbats


def simulate(
    model: model.Model,
    data: pd.DataFrame,
    atbats: pd.DataFrame | None = None,
    pitches: pd.DataFrame | None = None,
    test_year: int = 2016,
) -> pd.DataFrame:
    """Backtest a given model by simulating on one years worth of data.

    This function does the following:
    1. The given model is trained on all data with game_year < test_year.
    2. Data with game_year = test_year is split up and sorted by date.
    3. For each days worth of data, the model is used to predict the outcome for
        each batter starting that day.
    4. The data for that day is then incorporated into the model before moving
        on to the next day.

    :param model: a Model for prediction
    :param data: a data set of batter/game outcomes.
    :param atbats: optional dataset of atbats and outcomes.
    :param pitches: optional dataset of pitches and outcomes.
    :param test_year: season to do simulation for
    
    returns The predicted hit probability for each entry in data.
    """
    ab_train = atbats[atbats.game_year < test_year] if atbats is not None else None
    p_train = pitches[pitches.game_year < test_year] if pitches is not None else None
    if atbats is not None:
        ab_test = atbats[atbats.game_year == test_year]
        ab_groups = ab_test.groupby("game_date")
    if pitches is not None:
        p_test = pitches[pitches.game_year == test_year]
        p_groups = p_test.groupby("game_date")

    train = data[data.game_year < test_year]
    test = data[data.game_year == test_year]
    model.train(train, ab_train, p_train)

    results = []

    for date, group in test.groupby("game_date", observed=True):
        print(date, group.shape[0])
        tmp = group[["game_date", "batter", "hit"]].copy()
        tmp["proba"] = model.predict(group).values  # may contain same batter twice
        results.append(tmp)
        ab_group = ab_groups.get_group(date) if atbats is not None else None
        p_group = p_groups.get_group(date) if pitches is not None else None
        model.update(group, ab_group, p_group)

    return pd.concat(results)
