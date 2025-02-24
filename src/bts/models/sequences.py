import chex
import jax
from typing import Sequence
import pandas as pd
from bts.data import load
import numpy as np
import jax.numpy as jnp


def normalize_data(
    pitches: pd.DataFrame,
    key_columns: Sequence[str],
    categorical_columns: Sequence[str],
    numerical_columns: Sequence[str],
) -> pd.DataFrame:

    results = pitches[key_columns].copy()
    results[categorical_columns] = pitches[categorical_columns].astype("category")

    for col in categorical_columns:
        results[col] = results[col].cat.codes

    for col in numerical_columns:
        tmp = pitches[col]
        results[col] = (tmp - tmp.mean()) / tmp.std()

    return results


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


@chex.dataclass
class PitchInfoBlock:
    categorical: jax.Array  # SHAPE (B, S, F1)
    categorical_missing_mask: jax.Array  # SHAPE (B, S, F1)
    numerical: jax.Array  # SHAPE (B, S, F2)
    numerical_missing_mask: jax.Array  # SHAPE (B, S, F2)

    def metadata(self) -> tuple[list[int], int]:
        vocab_sizes = [int(x) for x in self.categorical.max(axis=(0, 1)) + 1]
        numerical_features = self.numerical.shape[-1]
        return vocab_sizes, numerical_features

    @classmethod
    def from_groups(
        cls,
        groups: pd.api.typing.DataFrameGroupBy,
        categorical_cols: Sequence[str],
        numerical_cols: Sequence[str],
        sequence_length: int = 400,
    ) -> "PitchInfoBlock":

        categorical = pad_or_truncate(
            groups[categorical_cols].apply(np.array).values,
            sequence_length,
            pad_value=-1,
        )
        numerical = pad_or_truncate(
            groups[numerical_cols].apply(np.array).values,
            sequence_length,
            pad_value=jnp.nan,
        )

        return PitchInfoBlock(
            categorical=categorical + 1,
            categorical_missing_mask=categorical != -1,
            numerical=jnp.nan_to_num(numerical),
            numerical_missing_mask=jnp.isnan(numerical),
        )


@chex.dataclass
class PitchSequences:
    pitch_context: PitchInfoBlock
    pitcher_outcomes: PitchInfoBlock
    batter_outcomes: PitchInfoBlock

    def __getitem__(self, key: str) -> PitchInfoBlock:
        match key: 
            case 'pitch_context':
                return self.pitch_context
            case 'pitcher_outcomes':
                return self.pitcher_outcomes
            case 'batter_outcomes':
                return self.batter_outcomes
        raise ValueError(f'Unrecognized {key=}')

    @property
    def num_sequences(self) -> int:
        return self.pitch_context.categorical.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.pitch_context.categorical.shape[1]

    @classmethod
    def load(
        cls,
        context_categorical: Sequence[str],
        context_numerical: Sequence[str],
        pitcher_categorical: Sequence[str],
        pitcher_numerical: Sequence[str],
        batter_categorical: Sequence[str],
        batter_numerical: Sequence[str],
        groupby_key: str = "pitcher",
        sequence_length: int = 400,
    ) -> "PitchSequences":

        key_cols = ["batter", "pitcher"]
        categorical_cols = context_categorical + pitcher_categorical + batter_categorical
        numerical_cols = context_numerical + pitcher_numerical + batter_numerical

        pitches = load.load_pitches(columns=key_cols+categorical_cols+numerical_cols)

        pitches = normalize_data(pitches, key_cols, categorical_cols, numerical_cols)

        groups = pitches.groupby(groupby_key, observed=True)

        context = PitchInfoBlock.from_groups(
            groups=groups,
            categorical_cols=context_categorical,
            numerical_cols=context_numerical,
            sequence_length=sequence_length,
        )

        pitcher = PitchInfoBlock.from_groups(
            groups=groups,
            categorical_cols=pitcher_categorical,
            numerical_cols=pitcher_numerical,
            sequence_length=sequence_length,
        )

        batter = PitchInfoBlock.from_groups(
            groups=groups,
            categorical_cols=batter_categorical,
            numerical_cols=batter_numerical,
            sequence_length=sequence_length,
        )

        return PitchSequences(
            pitch_context=context, pitcher_outcomes=pitcher, batter_outcomes=batter
        )

    def metadata(self) -> ...:
        seq_len = self.pitch_context.categorical.shape[1]
        result = {
            "pitch_context": self.pitch_context.metadata(),
            "pitcher_outcomes": self.pitcher_outcomes.metadata(),
            "batter_outcomes": self.batter_outcomes.metadata(),
        }
        return seq_len, result

    def sample(self, batch_size: int) -> "PitchSequences":
        num_sequences = self.pitch_context.categorical.shape[0]
        idx = np.random.choice(num_sequences, size=batch_size)
        return jax.tree.map(lambda x: x[idx], self)

