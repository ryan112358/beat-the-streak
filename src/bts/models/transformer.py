import jax
import jax.numpy as jnp
from flax import nnx
import optax
from bts.models import losses


class TransformerBlock(nnx.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 0,
    ):
        rngs = nnx.Rngs(seed)
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
            in_features=hidden_dim, out_features=4*hidden_dim, dtype=dtype, rngs=rngs
        )
        self.layernorm2 = nnx.LayerNorm(num_features=4*hidden_dim, dtype=dtype, rngs=rngs)
        self.linear2 = nnx.Linear(
            in_features=4*hidden_dim, out_features=hidden_dim, dtype=dtype, rngs=rngs
        )
        self.dropout = nnx.Dropout(
            rate=dropout_rate,
        )

    def __call__(self, x: jax.Array, mask=None, deterministic=False):
        """Compute forward activations through a transformer block.

        Args:
            x: An array of shape (B, S, H) where
                B is the batch size
                S is the sequence length
                H is the hidden dimensionality

        Returns:
            An array of shape (B, S, H)
        """

        x = x + self.attention(x, decode=False, mask=mask)
        x = self.dropout(self.layernorm(x))
        x2 = nnx.gelu(self.linear1(x))
        x2 = self.dropout(self.layernorm2(x2))
        x = x + nnx.gelu(self.linear2(x2))
        x = self.dropout(self.layernorm(x))
        return x


class Transformer(nnx.Module):

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
    ):
        # embeddings = hit / no hit
        rngs = nnx.Rngs(0)
        self.seq_len = seq_len
        self.type_embed = nnx.Embed(
            num_embeddings=vocab_size, features=hidden_dim, rngs=rngs
        )
        self.loc_embed = nnx.Linear(
            in_features=num_numerical_features, 
            out_features=hidden_dim, 
            dtype=dtype, 
            rngs=rngs
        )
        self.position_embed = nnx.Embed(
            num_embeddings=seq_len, features=hidden_dim, rngs=rngs
        )
        self.blocks = [
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
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
            rngs=rngs
        )
        self.numeric_final_means = nnx.Linear(
            in_features=hidden_dim, 
            out_features=mixture_components*num_numerical_features, 
            dtype=dtype, 
            rngs=rngs
        )
        self.numeric_final_stddevs = nnx.Linear(
            in_features=hidden_dim, 
            out_features=mixture_components*num_numerical_features, 
            dtype=dtype, 
            rngs=rngs
        )
 
    def __call__(self, 
                 pitch_types: jax.Array, 
                 pitch_locs: jax.Array,
                 mask=None, 
                 deterministic=False
    ) -> jax.Array:
        mask = nnx.make_causal_mask(pitch_types)
        x = (
                self.type_embed(pitch_types) + 
                # padded pitch locations are nan
                self.loc_embed(pitch_locs) +
                self.position_embed(jnp.arange(self.seq_len)[None])
        )
        for layer in self.blocks:
            x = layer(x, mask, deterministic)
        # TODO: can the mixture components depend on the observed pitch types?
        logits = self.type_final_logits(x)
        weights = nnx.softmax(self.numeric_final_weights(x))
        means = self.numeric_final_means(x)
        stddevs = nnx.relu(self.numeric_final_stddevs(x)) + 0.001
        return logits, weights, means, stddevs



@nnx.jit
def train_step(
    model: Transformer,
    optimizer: nnx.Optimizer,
    ptypes: jax.Array, 
    plocs: jax.Array,
    type_missing_mask: jax.Array,
    loc_missing_mask: jax.Array
) -> float:
    def loss_fn(model, ptypes, plocs, type_missing_mask, loc_missing_mask):
        logits, weights, means, stddevs = model(ptypes, plocs)

        ptypes = ptypes[:, 1:]
        plocs = plocs[:, 1:]
        type_missing_mask = type_missing_mask[:, 1:]
        loc_missing_mask = loc_missing_mask[:, 1:]

        logits = logits[:, :-1]
        weights = weights[:, :-1]
        means = means[:, :-1]
        stddevs = stddevs[:, :-1]

        type_loss = losses.masked_crossentropy(logits, ptypes, type_missing_mask)
        real_loss = losses.mixture_density_loss(weights, means, stddevs, plocs, loc_missing_mask)
        return 5*type_loss + real_loss, (type_loss, real_loss)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, ptypes, plocs, type_missing_mask, loc_missing_mask)
    optimizer.update(grads)
    return aux


