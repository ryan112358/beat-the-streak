import jax
import jax.numpy as jnp
import argparse
from bts.models import transformer, sequences, baseline
from bts.data import load
import pandas as pd
import numpy as np
from flax import nnx
import optax
import IPython
import orbax.checkpoint

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["seq_len"] = 4096
    params["hidden_dim"] = 64
    params["num_layers"] = 4
    params["num_heads"] = 4
    params["iterations"] = 1000
    params["batch_size"] = 1
    params["mixture_components"] = 8
    params["learning_rate"] = 1e-4
    params["checkpoint_path"] = None

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
    parser.add_argument(
        "--mixture_components", type=int, help="number of mixture components"
    )
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint path")

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

    """
    context_categorical = ['pitch_type']
    context_numerical = []
    pitcher_categorical = ['zone']
    pitcher_numerical = ["plate_x", "plate_z"]

    batter_categorical = [] #"description"]
    batter_numerical = [] #"estimated_ba_using_speedangle"]
    """

    data, pitches = sequences.PitchSequences.load(
        context_categorical,
        context_numerical,
        pitcher_categorical,
        pitcher_numerical,
        batter_categorical,
        batter_numerical,
        groupby_key="pitcher",
        sequence_length=args.seq_len,
    )

    baseline_results = baseline.independent_baseline(
        pitches,
        categorical_columns=pitcher_categorical + batter_categorical,
        numerical_columns=pitcher_numerical + batter_numerical,
    )
    print(baseline_results)

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

    def make_fresh_optimizer(model):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=args.learning_rate,
            warmup_steps=1000,
            decay_steps=10000,
        )

        optax_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(schedule, 0.9)
        )
        return nnx.Optimizer(model, optax_optimizer)

    optimizer = make_fresh_optimizer(model)

    columns = (
        ["total"]
        + pitcher_categorical
        + pitcher_numerical
        + batter_categorical
        + batter_numerical
    )
    results = pd.DataFrame(columns=columns, index=range(args.iterations))

    print("\t".join(columns))

    eval_every = 100
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    mixture_components = args.mixture_components
    for k in range(1):
        for t in range(args.iterations):
            batch = data.sample(args.batch_size)
            loss, aux1, aux2 = transformer.train_step(model, optimizer, batch)
            row = [loss] + list(aux1[0]) + list(aux1[1]) + list(aux2[0]) + list(aux2[1])
            results.loc[t] = [float(x) for x in row]
            if t % eval_every == eval_every - 1:
                metrics = results.loc[t - eval_every + 1 : t].mean()
                normalized = (
                    (metrics - baseline_results).reindex(metrics.index).map(np.exp)
                )
                print(normalized)
                if args.checkpoint_path:
                    # https://github.com/google/orbax/issues/1105
                    convert = lambda x: jax.random.key_data(x) if type(x) == type(jax.random.key(0)) else x
                    state = jax.tree.map(convert, nnx.state(model))

                    checkpointer.save(args.checkpoint_path, state)
                # print('\t'.join(['%.2f' % s for s in normalized.values]))

        num_layers *= 2
        hidden_dim *= 2
        num_heads *= 2
        mixture_components *= 2
        model = model.grow(
            num_layers, hidden_dim, num_heads, mixture_components, new_params_weight=0.1
        )
        optimizer = make_fresh_optimizer(model)

    # IPython.embed()

    results.to_csv("results.csv")
