from bts.evaluation.simulate import simulate, pretrain_model
from bts.evaluation import metrics, visualize
from bts.models.player_game.model import Baseline
from bts.models.player_game.parametric import Logistic, Singlearity
from bts.models.player_game.btsnet_v1 import BTSNet
from bts.models.player_game.sequential import EloSystem
import socket
import matplotlib.pyplot as plt
import argparse
from bts.data import load_data, load_atbats, load_pitches
import pandas as pd
import os
import pickle
from IPython import embed


def process_results(results, folder, stdout=True):
    """Compute metrics + visualizations for the results and dump to folder."""
    with open(folder + "/summary.txt", "w") as file:
        file.write(pick_distribution(results).head(15).to_string())
        file.write("\n\n\n")
        for metric, score_fn in metrics.ALL_METRICS.items():
            line = "%s: %.4f \n" % (metric, score_fn(results))
            file.write(line)
            if stdout:
                print(line)

    visualize.success_distribution(results)
    plt.savefig(folder + "/success_distribution.png")
    visualize.success_curve(results)
    plt.savefig(folder + "/success_curve.png")
    visualize.calibration_curve(results, bucket_sizes=[500])
    plt.savefig(folder + "/calibration_curve.png")
    plt.close('all')


def main(model, year):
    data = load_data()
    atbats = load_atbats()
    pitches = load_pitches()
    data["num_hits"] = data.hit
    data['hit'] = data.hit >= 1

    train = data[data.game_year < year]
    test = data[data.game_year >= year]

    model.train(train)
    base = "./"
    folder = base + str(model)
    if not os.path.exists(folder):
        os.mkdir(folder)

    results = []
    for date, group in test.groupby('game_date', observed=True):
        print(date)
        tmp = group[['game_date', 'batter', 'hit']].copy()
        tmp['proba'] = model.predict(group).values
        tmp["game_date"] = tmp.game_date.astype(str).str[:10]
        results.append(tmp)
        model.update(group)

        output = pd.concat(results)
        output.to_csv(folder + "/results.csv", index=False)

    process_results(output, folder)


def pick_distribution(results, top=2):
    topk = results.groupby("game_date").proba.nlargest(top)
    idx = topk.index.get_level_values(1)
    groups = results.loc[idx].groupby("batter", observed=True).hit
    ans = {"picks": groups.count(), "success": groups.mean()}
    return pd.concat(ans, axis=1, sort=False).sort_values("picks", ascending=False)


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["year"] = 2018
    params["model"] = "baseline"

    return params


if __name__ == "__main__":

    models = {
        "baseline": Baseline(),
        "logistic": Logistic(),
        "elo": EloSystem(),
        "singlearity": Singlearity(0.001),
        "btsnet": BTSNet(log=True),
    }

    description = "simulate BTS model"
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--year", type=int, help="year to simulate")
    parser.add_argument("--model", choices=models.keys(), help="model to use")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    main(models[args.model], args.year)
