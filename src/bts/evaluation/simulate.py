import socket
if 'swarm' in socket.gethostname():
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from bts.evaluation import metrics, visualize
import pandas as pd
import os
import pickle

def simulate(model, data, atbats = None, pitches = None, test_year = 2016):
    """
    :param model: a Model for prediction
    :param data: a data set
    :param test_year: season to do simulation for
    """
    ab_train = atbats[atbats.year < test_year] if atbats is not None else None
    p_train = pitches[pitches.year < test_year] if pitches is not None else None
    if atbats is not None:
        ab_test = atbats[atbats.year == test_year]
        ab_groups = ab_test.groupby('game_date')
    if pitches is not None:
        p_test = pitches[pitches.year == test_year]
        p_groups = p_test.groupby('game_date')
   
    train = data[data.year < test_year]
    test = data[data.year == test_year]
    model.train(train, ab_train, p_train)

    results = []
 
    for date, group in test.groupby('game_date', observed=True):
        print(date, group.shape[0])
        tmp = group[['game_date', 'batter', 'hit']].copy()
        #tmp['proba'] = model.predict(group)[tmp.batter].values
        tmp['proba'] = model.predict(group).values # may contain same batter twice
        results.append(tmp)
        ab_group = ab_groups.get_group(date) if atbats is not None else None
        p_group = p_groups.get_group(date) if pitches is not None else None
        model.update(group, ab_group, p_group)

    return pd.concat(results)

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
    params['year'] = 2016
    params['model'] = 'baseline'

    return params

def main(model, year):
    stuff = load()
    data = stuff['data']
    atbats = stuff['atbats']
    pitches = stuff['pitches']

    base = os.environ['BTSRESULTS']
    folder = base + '/' + str(model) + '_' + str(year)
    if not os.path.exists(folder):
        os.mkdir(folder)

    try:
        results = pd.read_csv(folder + '/results.csv', index_col=0)
    except:
        results = simulate(model, data, atbats, pitches, test_year = year)
        results.to_csv(folder + '/results.csv')

    # only works for LogisticRegression
    #summary = model.summary()
    #if summary is not None:
    #    with open(folder + '/summary.pkl', 'wb') as file:
    #        pickle.dump(summary, file)

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

if __name__ == '__main__':

    
    #ensemble = model.Ensemble([model.Baseline(), sequential.EloSystem(), parametric.Logistic()], weights=[0.65, 0.17, 0.18])

    #comp = composed.Compose(atbats.BaselineAB(), plateapps.OraclePA())

    models = { 
        'baseline' : model.Baseline(), 
        'exp' : sequential.ExpSmoothing(), 
        'elo' : sequential.EloSystem(), 
        #'logit' : parametric.Logistic(), 
        #'ensemble' : ensemble, 
        #'composed' : comp,
        #'xgboost' : parametric.XGBoost(),
        #'lgbm' : parametric.LightGBM(),
        #'automl' : parametric.AutoML()
 }
    
    description = 'simulate BTS model'
    formatter = argparse.ArgumentDefaultsHelpFormatter    
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--year', type=int, help='year to simulate')
    parser.add_argument('--model', choices=models.keys(), help='model to use')

    parser.set_defaults(**default_params()) 
    args = parser.parse_args()
    
    main(models[args.model], args.year)
