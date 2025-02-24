import jax
import jax.numpy as jnp
import argparse
from bts.models import transformer, sequences
from bts.data import load
import pandas as pd
import numpy as np
from flax import nnx
import optax
import IPython

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["seq_len"] = 400
    params["hidden_dim"] = 64
    params["num_layers"] = 4
    params["num_heads"] = 4
    params["iterations"] = 1000
    params["batch_size"] = 8
    params["mixture_components"] = 8
    params["sequence_length"] = 400

    return params


if __name__ == "__main__":
    description = "Train a transformer to predict the next pitch"
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--seq_len", type=int, help="sequence length")
    parser.add_argument("--hidden_dim", type=int, help="hidden dimensionality")
    parser.add_argument("--num_layers", type=int, help="number of layers")
    parser.add_argument("--num_heads", type=int, help="number of heads")
    parser.add_argument("--iterations", type=int, help="number of iterations")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--sequence_length", type=int, help="sequence length")
    parser.add_argument(
        "--mixture_components", type=int, help="number of mixture components"
    )

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    context_categorical = ["pitch_number"]
    context_numerical = []

    pitcher_categorical = ["pitch_type", "zone"]
    pitcher_numerical = [
        "release_speed",
        "release_pos_x",
        "release_pos_z",
        "spin_axis",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "effective_speed",
        "release_spin_rate",
        "release_extension",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
    ]

    batter_categorical = ["description", "hit_location", "launch_speed_angle", "events"]
    batter_numerical = [
        "hc_x",
        "hc_y",
        "hit_distance_sc",
        "launch_speed",
        "launch_angle",
        "spray_angle",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "babip_value",
        "iso_value",
    ]

    pitcher_categorical = ["pitch_type"]
    pitcher_numerical = ["plate_x", "plate_z"]

    batter_categorical = ["description"]
    batter_numerical = ["estimated_ba_using_speedangle"]

    data = sequences.PitchSequences.load(
        context_categorical,
        context_numerical,
        pitcher_categorical,
        pitcher_numerical,
        batter_categorical,
        batter_numerical,
        groupby_key="pitcher",
        sequence_length=args.sequence_length,
    )

    model = transformer.Transformer(
        sequence_metadata=data.metadata(),
        hidden_dim=args.hidden_dim,
        mixture_components=args.mixture_components,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    params = nnx.state(model, nnx.Param)
    total_params = sum(x.size for x in jax.tree.leaves(params))
    print("Model Size: %.2f M" % (total_params / 10**6))

    optimizer = nnx.Optimizer(model, optax.adamw(1e-4, 0.9))
    pitch_cat_loss = pitch_num_loss = bat_cat_loss = bat_num_loss = 0
    type_loss = real_loss = 0.0
    for t in range(args.iterations):
        aux = transformer.train_step(model, optimizer, data.sample(args.batch_size))
        pitch_cat_loss += sum(aux[0][0])
        pitch_num_loss += aux[0][1]
        bat_cat_loss += sum(aux[1][0])
        bat_num_loss += aux[1][1]
        if t % 100 == 99:
            print(
                pitch_cat_loss / 100,
                pitch_num_loss / 100,
                bat_cat_loss / 100,
                bat_num_loss / 100,
            )
            pitch_cat_loss = pitch_num_loss = bat_cat_loss = bat_num_loss = 0.0
