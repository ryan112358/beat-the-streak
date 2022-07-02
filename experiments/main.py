from bts.evaluation.simulate import simulate
from bts.evaluation import metrics, visualize
from bts.models.player_game.model import Baseline
from bts.models.player_game.parametric import Logistic
from bts.models.player_game.sequential import EloSystem
import socket
if 'swarm' in socket.gethostname():
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from bts.data import load_data, load_atbats, load_pitches
import pandas as pd
import os
import pickle

def main(model, year):
    data = load_data()
    atbats = load_atbats()
    pitches = load_pitches()

    base = './' #os.environ['BTSRESULTS']
    folder = base + str(model) + '_' + str(year)
    if not os.path.exists(folder):
        os.mkdir(folder)

    #try:
    #    results = pd.read_csv(folder + '/results.csv', index_col=0)
    #except:
    results = simulate(model, data, atbats, pitches, test_year = year)
    results['game_date'] = results.game_date.astype(str).str[:10]
    results.to_csv(folder + '/results.csv', index=False)

    with open(folder + '/summary.txt', 'w') as file:
        file.write(pick_distribution(results).head(15).to_string())
        file.write('\n\n\n')
        for metric, score in metrics.all_metrics().items():
            line = '%s: %.4f \n' % (metric, score(results))
            file.write(line)
            print(line)

    visualize.success_distribution(results)
    plt.savefig(folder + '/success_distribution.png')
    visualize.success_curve(results)
    plt.savefig(folder + '/success_curve.png')
    visualize.calibration_curve(results, bucket_size=250)
    plt.savefig(folder + '/calibration_curve.png')

    return results

def pick_distribution(results, top=2):
    topk = results.groupby('game_date').proba.nlargest(top)
    idx = topk.index.get_level_values(1)
    counts = results.loc[idx].batter.value_counts()
    success = results.loc[idx].groupby('batter').hit.mean(numeric_only=False)
    ans = pd.concat([counts, success], axis=1, sort=False).sort_values('batter', ascending=False)
    return ans

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

    models = { 'baseline': Baseline(), 'logistic': Logistic(), 'elo': EloSystem() }

    description = 'simulate BTS model'
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--year', type=int, help='year to simulate')
    parser.add_argument('--model', choices=models.keys(), help='model to use')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    main(models[args.model], args.year)
