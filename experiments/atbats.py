from bts.evaluation.simulate_atbats import simulate
from bts.evaluation import metrics
from bts.models.atbat.model import Baseline
from bts.models.atbat.parametric import Logistic
from bts.models.atbat.torch import TabNet, NeuralNet
#from bts.models.atbat.sequential import EloSystem
import argparse
from bts.data import load_data, load_atbats, load_pitches
import pandas as pd
import os
import pickle

def main(model, year):
    #data = load_data()
    atbats = load_atbats()
    pitches = load_pitches()

    results = simulate(model, atbats, pitches, test_year = year)
    results['game_date'] = results.game_date.astype(str).str[:10]

    print('Likelihood', metrics.likelihood(results))

    return results

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['year'] = 2018
    params['model'] = 'baseline'

    return params

if __name__ == '__main__':

    models = { 'baseline': Baseline(), 'logistic': Logistic(), 'tabnet': TabNet(), 'nnet' : NeuralNet() }

    description = 'simulate BTS model'
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--year', type=int, help='year to simulate')
    parser.add_argument('--model', choices=models.keys(), help='model to use')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    main(models[args.model], args.year)
