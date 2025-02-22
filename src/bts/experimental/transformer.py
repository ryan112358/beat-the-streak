import jax
import jax.numpy as jnp
import argparse
from bts.models import transformer
from bts.data import load
import pandas as pd
import numpy as np
from flax import nnx
import optax

#jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_enable_x64", True)

NUMERICAL_COLUMNS = [
    'release_speed', 
    'release_pos_x', 
    'release_pos_z', 
    # 'spin_dir', all null
    # 'zone', categorical
    'spin_axis',
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
    'effective_speed',
    'release_spin_rate',
    'release_extension',
    'pfx_x',
    'pfx_z',
    'plate_x',
    'plate_z'
]

def pad_or_truncate(arrays, m, pad_value):
    # Pads or truncates a list of arrays of a_1, ..., a_n of size m_i x p.
    n = len(arrays)
    p = arrays[0].shape[1]
    result = np.full((n, m, p), pad_value)

    for i, arr in enumerate(arrays):
        rows = arr.shape[0]
        if rows <= m:
            result[i, :rows, :] = jnp.nan_to_num(arr, pad_value)
        else:
            result[i, :m, :] = jnp.nan_to_num(arr[:m, :], pad_value)

    return result


def load_sequences():
    pitches = load.load_pitches(columns=['pitcher', 'pitch_type'] + NUMERICAL_COLUMNS)
    # this allows plate_x and plate_y to be nan
    subset = pitches.dropna(subset='pitch_type')

    # normalize numerical data, subject to change. 
    tmp = subset[NUMERICAL_COLUMNS]
    subset.loc[:,NUMERICAL_COLUMNS] = (tmp - tmp.mean()) / tmp.std()

    ptypes = subset.groupby('pitcher', observed=True)['pitch_type'].apply(lambda g: np.array(g.cat.codes.values)[:,None])
    plocs = subset.groupby('pitcher', observed=True)[NUMERICAL_COLUMNS].apply(np.array)

    ptypes = pad_or_truncate(ptypes.values, 400, pad_value=-1)[:,:,0]
    plocs = pad_or_truncate(plocs.values, 400, pad_value=-128)

    type_missing_mask = ptypes != -1
    loc_missing_mask = plocs != -128

    return ptypes, plocs, type_missing_mask, loc_missing_mask


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
    params["batch_size"] = 64
    params["mixture_components"] = 8

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
    parser.add_argument("--mixture_components", type=int, help="number of mixture components")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    ptypes, plocs, type_missing_mask, loc_missing_mask = load_sequences()

    model = transformer.Transformer(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_numerical_features=plocs.shape[-1],
        mixture_components=args.mixture_components,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=ptypes.max() + 1,
    )

    optimizer = nnx.Optimizer(model, optax.adamw(1e-4, 0.9))
    BATCH_SIZE = args.batch_size
    type_loss = real_loss = 0
    for t in range(args.iterations):
        idx = np.random.choice(ptypes.shape[0], size=BATCH_SIZE)
        tl, rl = transformer.train_step(
                model, 
                optimizer, 
                ptypes[idx], 
                plocs[idx],
                type_missing_mask[idx],
                loc_missing_mask[idx],
        )
        type_loss += tl; real_loss += rl
        if t % 100 == 99:
            print(type_loss / 100, real_loss / 100)
            type_loss = real_loss = 0
