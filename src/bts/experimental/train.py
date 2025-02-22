import jax
import jax.numpy as jnp
import argparse
from bts.models import transformer, losses
from bts.data import load
import pandas as pd
import numpy as np
from flax import nnx
import optax
import IPython

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

PITCH_CONTEXT = ['pitch_number']

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


def load_sequences() -> losses.InputData:
    columns = ['pitcher', 'pitch_type'] + NUMERICAL_COLUMNS + PITCH_CONTEXT
    pitches = load.load_pitches(columns=columns)
    # this allows plate_x and plate_y to be nan
    subset = pitches.dropna(subset='pitch_type')

    # normalize numerical data, subject to change. 
    tmp = subset[NUMERICAL_COLUMNS]
    subset.loc[:,NUMERICAL_COLUMNS] = (tmp - tmp.mean()) / tmp.std()

    groups = subset.groupby('pitcher', observed=True)

    ptypes = groups['pitch_type'].apply(lambda g: np.array(g.cat.codes.values)[:,None])
    plocs = groups[NUMERICAL_COLUMNS].apply(np.array)
    pitch_in_atbat = groups['pitch_number'].apply(lambda x: np.array(x-1)[:,None])

    # only padded values should be -1 here.  There should be no other missing values
    ptypes = pad_or_truncate(ptypes.values, 400, pad_value=-1)[:,:,0]
    pitch_in_atbat = pad_or_truncate(pitch_in_atbat.values, 400, pad_value=-1)[:,:,0]
    plocs = pad_or_truncate(plocs.values, 400, pad_value=-128)

    type_missing_mask = ptypes != -1
    loc_missing_mask = plocs != -128


    return losses.InputData(
        pitch_types=ptypes,
        pitch_features=plocs,
        type_missing_mask=type_missing_mask,
        feature_missing_mask=loc_missing_mask,
        pitch_in_atbat=pitch_in_atbat
    )

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
    params["batch_size"] = 8
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

    data = load_sequences()

    model = transformer.Transformer(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_numerical_features=data.pitch_features.shape[-1],
        mixture_components=args.mixture_components,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=data.pitch_types.max() + 1,
    )

    params = nnx.state(model, nnx.Param)
    total_params = sum(x.size for x in jax.tree.leaves(params))
    print('Model Size: %.2f M' % (total_params / 10**6))


    optimizer = nnx.Optimizer(model, optax.adamw(1e-4, 0.9))
    type_loss = real_loss = 0
    for t in range(args.iterations):
        tl, rl = transformer.train_step(
                model, 
                optimizer, 
                data.sample(args.batch_size)
        )
        type_loss += tl; real_loss += rl
        if t % 100 == 99:
            print(type_loss / 100, real_loss / 100)
            type_loss = real_loss = 0
