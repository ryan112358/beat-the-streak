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
from bts.models import losses


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
            qkv_features=hidden_dim // num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype,
            rngs=rngs,
        )
        self.layernorm = nnx.LayerNorm(num_features=hidden_dim, dtype=dtype, rngs=rngs)
        self.linear1 = nnx.Linear(
            in_features=hidden_dim, out_features=4 * hidden_dim, dtype=dtype, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=4 * hidden_dim, out_features=hidden_dim, dtype=dtype, rngs=rngs
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

        x = x + self.attention(x, decode=False, mask=mask, deterministic=deterministic)
        x = self.layernorm(x)
        x = self.dropout(x, deterministic=deterministic)

        x = x + self.linear2(nnx.gelu(self.linear1(x)))
        x = self.layernorm(x)
        x = self.dropout(x, deterministic=deterministic)

        return x


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
        seq_len: int,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        num_numerical_features: int,
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
            vocab_size: The size of the vocabulary for categorical input features (e.g., pitch types).
            num_numerical_features: The number of continuous numerical input features (e.g., pitch locations).
            mixture_components: The number of components in the mixture density network for numerical feature prediction.
            dropout_rate: Dropout probability.
            dtype: Data type for parameters and computations.
            seed: Random seed for initialization.
        """
        rngs = nnx.Rngs(seed)
        self.seq_len = seq_len
        self.type_embed = nnx.Embed(
            num_embeddings=vocab_size, features=hidden_dim, rngs=rngs
        )
        self.loc_embed = nnx.Linear(
            in_features=num_numerical_features,
            out_features=hidden_dim,
            dtype=dtype,
            rngs=rngs,
        )
        self.pitch_in_ab_embed = nnx.Embed(
            num_embeddings=10, features=hidden_dim, rngs=rngs
        )
        self.position_embed = nnx.Embed(
            num_embeddings=seq_len, features=hidden_dim, rngs=rngs
        )
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
        self.type_final_logits = nnx.Linear(
            in_features=hidden_dim, out_features=vocab_size, dtype=dtype, rngs=rngs
        )
        self.numeric_final_weights = nnx.Linear(
            in_features=hidden_dim,
            out_features=mixture_components,
            dtype=dtype,
            rngs=rngs,
        )
        self.numeric_final_means = nnx.Linear(
            in_features=hidden_dim,
            out_features=mixture_components * num_numerical_features,
            dtype=dtype,
            rngs=rngs,
        )
        self.numeric_final_variances = nnx.Linear(
            in_features=hidden_dim,
            out_features=mixture_components * num_numerical_features,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        batch: losses.InputData,
        mask=None,
        deterministic=False,
    ) -> losses.OutputDistribution:
        """Compute forward pass for the Transformer model.

        Args:
            batch: The input data, including pitch types, features, and context.
            mask: Optional attention mask, will be a causal mask if not provided.
            deterministic: Whether to disable dropout.

        Returns:
            The predicted output distribution for each pitch in the sequence. 
        """
        if mask is None:
            mask = nnx.make_causal_mask(batch.pitch_types)
        x = (
            self.type_embed(batch.pitch_types)
            + self.loc_embed(batch.pitch_features)
            + self.position_embed(jnp.arange(self.seq_len)[None])
            #+ self.pitch_in_ab_embed(jnp.minimum(batch.pitch_in_atbat, 9))
        )
        for layer in self.blocks:
            x = layer(x, mask, deterministic)

        # TODO: can the mixture components depend on the observed pitch types?
        return losses.OutputDistribution(
            logits = self.type_final_logits(x),
            mixture_weights = nnx.softmax(self.numeric_final_weights(x)),
            mixture_means = self.numeric_final_means(x),
            mixture_variances = nnx.relu(self.numeric_final_variances(x)) + 0.001
        )

@nnx.jit
def train_step(
    model: Transformer,
    optimizer: nnx.Optimizer,
    batch: losses.InputData,
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
        return losses.loss_fn(model(batch), batch)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    optimizer.update(grads)
    return aux
