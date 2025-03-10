"""Transformer models specialized for next-pitch prediction.

Notation:
    - B = BATCH_SIZE
    - S = SEQUENCE_LENGTH
    - F = NUMERICAL_FEATURES
    - M = MIXTURE_COMPONENTS 
    - V = VOCAB_SIZE (number of possible pitch types)
    - H = HIDDEN_DIMENSIONALITY
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from bts.models import losses, sequences
import functools


def causal_flash_attention_fn(query, key, value, **kwargs):
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        implementation="xla",
        is_causal=True,
        # local_window_size=1024
    )


def pad_and_fill(
    small_array: jax.Array, large_array: jax.Array, init_weight: float = 0.0
) -> jax.Array:
    small_shape = small_array.shape
    large_shape = large_array.shape
    if large_array.dtype == jnp.float32:
        large_array = large_array * init_weight

    if len(small_shape) != len(large_shape):
        raise ValueError("Arrays must have the same number of dimensions.")

    for i in range(len(small_shape)):
        if small_shape[i] > large_shape[i]:
            raise ValueError(
                f"Dimension {i} of small_array must be <= dimension {i} of large_array."
            )

    slices = tuple(slice(0, dim) for dim in small_shape)

    result = large_array.at[slices].set(small_array)
    return result


def sinusoidal_position_embeddings(S: int, H: int) -> jax.Array:
    """Returns a (1, S, H) array of positional embeddings."""
    position = jnp.arange(S)[:, None]
    indices = jnp.arange(H // 2)
    indices = 10000 ** (-2 * indices / H)
    embeddings = position * indices
    embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
    return embeddings[None]


def interleave_sequences(*arrays: jax.Array) -> jax.Array:
    """Interleave arrays of shape (B, S, H) along the sequence dimension."""
    B, S, H = arrays[0].shape
    n = len(arrays)
    reshaped_arrays = [arr.reshape(B, S, 1, H) for arr in arrays]
    interleaved = jnp.concatenate(reshaped_arrays, axis=2)
    return interleaved.reshape(B, n * S, H)


class FFNBlock(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
    ):

        self.linear1 = nnx.Linear(
            in_features=hidden_dim, out_features=4 * hidden_dim, dtype=dtype, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=4 * hidden_dim, out_features=hidden_dim, dtype=dtype, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.linear2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class TransformerBlock(nnx.Module):
    """A single Transformer block layer.

    Implements a standard Transformer block with:
    1. Multi-Head Self-Attention
    2. Layer Normalization
    3. Dropout
    4. Feed-Forward Network (Linear-GELU-Linear)
    5. Layer Normalization
    6. Dropout
    Residual connections are used around both the attention and FFN sub-layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initializes a TransformerBlock.

        Args:
            hidden_dim: The dimensionality of the input and output features.
            num_heads: The number of attention heads. Must divide `hidden_dim` evenly.
            dropout_rate: Dropout probability.
            dtype: Data type for parameters and computations.
            rngs: PRNGKey for parameter initialization.
        """
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
            rngs=rngs,
            attention_fn=causal_flash_attention_fn,
        )
        # self.layernorm1 = nnx.LayerNorm(num_features=hidden_dim, dtype=dtype, rngs=rngs)
        self.rmsnorm1 = nnx.RMSNorm(
            num_features=hidden_dim, dtype=dtype, rngs=rngs, use_scale=False
        )

        self.ffn = FFNBlock(
            hidden_dim=hidden_dim, dtype=dtype, rngs=rngs, dropout_rate=dropout_rate
        )

        # self.layernorm2 = nnx.LayerNorm(num_features=hidden_dim, dtype=dtype, rngs=rngs)
        self.rmsnorm2 = nnx.RMSNorm(
            num_features=hidden_dim, dtype=dtype, rngs=rngs, use_scale=False
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, mask=None, deterministic=False):
        """Compute forward activations through a transformer block.

        Args:
            x: An array of shape (B, S, H)
            mask: An optional attention mask of shape (B, 1, S, S).
                `mask[i, j]` is 1 when attention from token `i` to token `j` is allowed,
                and 0 otherwise.  For causal attention, use `nnx.make_causal_mask(x)`.
            deterministic: Whether to disable dropout. Useful for inference.

        Returns:
            An array of shape (B, S, H) representing the output of the transformer block.
        """

        # Attention requires ~ B * S^2 * H space, by using remat we do not have to pay an extra factor of L.
        # TODO: may not need to have mask here, hardcoding causal attention.
        attn = jax.remat(
            lambda x, mask: self.attention(
                x, decode=False, mask=mask, deterministic=deterministic
            )
        )

        y = self.rmsnorm1(x)
        y = attn(y, mask)
        y = self.dropout(y)
        y = x + y

        z = self.rmsnorm2(y)
        z = self.ffn(z, deterministic=deterministic)

        return y + z


class PitchEmbedding(nnx.Module):
    def __init__(
        self,
        sequence_metadata: ...,
        hidden_dim: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        seq_len, metadata = sequence_metadata
        categorical_embedding_dim = 8

        # last dimensions should be hidden_dim, but we set it to 1
        # so it can be broadcasted even if hidden_dim changes.
        dummy_layer = lambda x: jnp.zeros(x.shape[:-1] + (1,))

        self.embeddings = {}
        for col in metadata:
            vocab_sizes, numerical_features = metadata[col]
            print(col, vocab_sizes, numerical_features)

            categorical = [
                nnx.Embed(
                    num_embeddings=v, features=categorical_embedding_dim, rngs=rngs
                )
                for v in vocab_sizes
            ]

            in_features = 2 * numerical_features + len(vocab_sizes) * (
                categorical_embedding_dim + 1
            )
            linear = (
                nnx.Linear(
                    in_features=in_features,
                    out_features=hidden_dim,
                    dtype=dtype,
                    rngs=rngs,
                )
                if in_features > 0
                else dummy_layer
            )

            self.embeddings[col] = categorical, linear

    def __call__(self, batch: sequences.PitchSequences) -> jax.Array:
        xs = {}
        for key in ["pitch_context", "pitcher_outcomes", "batter_outcomes"]:
            embed_cat, linear = self.embeddings[key]

            info = batch[key]

            feats = [
                info.numerical,
                info.categorical_missing_mask,
                info.numerical_missing_mask,
            ]
            for embed_cat1, codes in zip(
                embed_cat, jnp.moveaxis(info.categorical, -1, 0)
            ):
                feats.append(embed_cat1(codes))
            feats = jnp.concatenate(feats, axis=2)

            xs[key] = linear(feats)

        # TODO: replace this with a date-based embedding
        # xs["pitch_context"] += sinusoidal_position_embeddings(
        #    batch.sequence_length, xs["pitch_context"].shape[-1]
        # )

        return xs


class OutputHead(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        mixture_components: int,
        vocab_sizes: list[int],
        num_numerical_features: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.mixture_components = mixture_components
        self.num_numerical_features = num_numerical_features

        dummy_layer = lambda x: jnp.zeros(x.shape[:-1] + (0,))

        self.categorical_final = [
            nnx.Linear(in_features=hidden_dim, out_features=v, dtype=dtype, rngs=rngs)
            for v in vocab_sizes
        ]

        self.numeric_final_weights = nnx.Linear(
            in_features=hidden_dim,
            out_features=mixture_components,
            dtype=dtype,
            rngs=rngs,
        )  # if num_numerical_features > 0 else dummy_layer

        self.numeric_final_means = (
            nnx.Linear(
                in_features=hidden_dim,
                out_features=mixture_components * num_numerical_features,
                dtype=dtype,
                rngs=rngs,
            )
            if num_numerical_features > 0
            else dummy_layer
        )

        self.numeric_final_variances = (
            nnx.Linear(
                in_features=hidden_dim,
                out_features=mixture_components * num_numerical_features,
                dtype=dtype,
                rngs=rngs,
            )
            if num_numerical_features > 0
            else dummy_layer
        )

    def __call__(self, x: jax.Array) -> losses.OutputDistribution:

        F = self.num_numerical_features
        M = self.mixture_components

        mixture_weights = nnx.softmax(self.numeric_final_weights(x))
        mixture_means = self.numeric_final_means(x).reshape(*x.shape[:2], M, F)
        mixture_variances = nnx.relu(self.numeric_final_variances(x)) + 0.001
        mixture_variances = mixture_variances.reshape(*x.shape[:2], M, F)
        mixture_cov = jax.vmap(jax.vmap(jax.vmap(jnp.diag)))(mixture_variances)

        return losses.OutputDistribution(
            logits=[layer(x) for layer in self.categorical_final],
            # TODO: can the mixture components depend on the observed pitch types?
            mixture_weights=mixture_weights,
            mixture_means=mixture_means,
            mixture_cov=mixture_cov,
        )


class Transformer(nnx.Module):
    """A Transformer model for sequence-to-sequence tasks.

    This Transformer model is designed for predicting both categorical (pitch type)
    and continuous (pitch location) outputs. It uses a stack of Transformer blocks
    followed by separate output layers for each prediction task.

    The input is assumed to be arranged in a meaningful sequential format.  E.g.,
    sequences of pitches thrown by a pitcher, ordered by date.
    """

    def __init__(
        self,
        sequence_metadata: ...,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        mixture_components: int,
        dropout_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 0,
    ):
        """Initializes a Transformer model.

        Args:
            seq_len: The maximum sequence length the model can handle.
            num_layers: The number of Transformer blocks in the encoder.
            hidden_dim: The dimensionality of the hidden layers and embeddings.
            num_heads: The number of attention heads in each Transformer block.
            vocab_size: The number of possible pitch types.
            num_numerical_features: The number of continuous numerical input features.
            mixture_components: The number of components in the mixture density network.
            dropout_rate: Dropout probability.
            dtype: Data type for parameters and computations.
            seed: Random seed for initialization.
        """
        self.sequence_metadata = sequence_metadata
        self.rngs = rngs = nnx.Rngs(seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mixture_components = mixture_components
        self.dropout_rate = dropout_rate
        self.dtype = dtype

        self.embedding = PitchEmbedding(sequence_metadata, hidden_dim, rngs, dtype)

        self.blocks = [
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                rngs=rngs,
                dropout_rate=dropout_rate,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        vocab_sizes, num_features = sequence_metadata[1]["pitcher_outcomes"]
        self.pitcher_head = OutputHead(
            hidden_dim=self.hidden_dim,
            mixture_components=mixture_components,
            vocab_sizes=vocab_sizes,
            num_numerical_features=num_features,
            rngs=rngs,
            dtype=dtype,
        )

        vocab_sizes, num_features = sequence_metadata[1]["batter_outcomes"]
        self.batter_head = OutputHead(
            hidden_dim=self.hidden_dim,
            mixture_components=mixture_components,
            vocab_sizes=vocab_sizes,
            num_numerical_features=num_features,
            rngs=rngs,
            dtype=dtype,
        )

    def grow(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        mixture_components: int,
        new_params_weight: float = 0.0,
    ) -> "Transformer":
        curr_layers = len(self.blocks)
        for _ in range(num_layers - curr_layers):
            block = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                rngs=self.rngs,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )
            nnx.update(block, jax.tree.map(jnp.zeros_like, nnx.state(block)))
            self.blocks.append(block)

        new = Transformer(
            sequence_metadata=self.sequence_metadata,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mixture_components=mixture_components,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )

        foo = functools.partial(pad_and_fill, init_weight=new_params_weight)

        new_state = jax.tree.map(foo, nnx.state(self), nnx.state(new))
        ratio = (hidden_dim // num_heads) // (self.hidden_dim // self.num_heads)

        ratio2 = hidden_dim // self.hidden_dim

        for i in range(num_layers):
            # This normalization assumes RMSNorm is used immediately
            # before the attention layer and the ffn layer
            leaf = new_state.blocks[i].attention.key["kernel"]
            leaf.value = leaf.value * ratio**0.25 / jnp.sqrt(ratio2)
            leaf = new_state.blocks[i].attention.query["kernel"]
            leaf.value = leaf.value * ratio**0.25 / jnp.sqrt(ratio2)
            leaf = new_state.blocks[i].attention.value["kernel"]
            leaf.value = leaf.value / jnp.sqrt(ratio2)

            leaf = new_state.blocks[i].ffn.linear1.kernel
            leaf.value = leaf.value / jnp.sqrt(ratio2)

        nnx.update(new, new_state)

        return new

    def __call__(
        self,
        batch: sequences.PitchSequences,
        mask=None,
        deterministic=False,
    ) -> losses.OutputDistribution:
        """Compute forward pass for the Transformer model

        Args:
            batch: The input data, including pitch types, features, and context.
            mask: Optional attention mask, will be a causal mask if not provided.
            deterministic: Whether to disable dropout.

        Returns:
            The predicted output distribution for each pitch in the sequence.
        """
        if mask is None:
            B, S = batch.num_sequences, batch.sequence_length
            mask = nnx.make_causal_mask(jax.ShapeDtypeStruct((B, S), jnp.float32))

        xs = self.embedding(batch)
        x = xs["pitch_context"] + xs["pitcher_outcomes"] + xs["batter_outcomes"]

        # the pitch outcome can depend on the context of the current pitch
        # as well as the context and outcome of all previous pitches.
        # x = interleave_sequences(
        #    xs["pitch_context"], xs["pitcher_outcomes"], xs["batter_outcomes"]
        # )

        for layer in self.blocks:
            x = layer(x, mask, deterministic)

        # [1,2,3,4,5] --> [2,3,4,5,1]
        shift = lambda tree: jax.tree.map(lambda x: jnp.roll(x, -1, axis=1), tree)

        x = x + shift(xs["pitch_context"])
        pitcher_outcome_prediction = self.pitcher_head(x)

        x = x + shift(xs["pitcher_outcomes"])
        batter_outcome_prediction = self.batter_head(x)
        return pitcher_outcome_prediction, batter_outcome_prediction


@nnx.jit
def train_step(
    model: Transformer,
    optimizer: nnx.Optimizer,
    batch: sequences.PitchSequences,
) -> tuple[float, float]:
    """Performs a single training step.

    This function defines the training step, including loss calculation,
    gradient computation, and optimizer update.

    Args:
        model: The Transformer model.
        optimizer: The Optax optimizer.
        batch: Batch of pitches.

    Returns:
        aux: Auxiliary information from the loss function (e.g., type_loss, real_loss).
    """

    def loss_fn(model, batch):
        pitcher_distribution, batter_distribution = model(batch)
        loss1, aux1 = pitcher_distribution.loss_fn(batch.pitcher_outcomes)
        loss2, aux2 = batter_distribution.loss_fn(batch.batter_outcomes)
        loss = (loss1 + loss2) / batch.tokens

        return loss, (loss, aux1, aux2)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)

    optimizer.update(grads)
    return aux
