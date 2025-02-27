"""Loss functions specialized for transformer-based next-pitch prediction.

Notation:
    - B = BATCH_SIZE
    - S = SEQUENCE_LENGTH
    - F = NUMERICAL_FEATURES
    - M = MIXTURE_COMPONENTS 
    - V = VOCAB_SIZE (number of possible pitch types)
    - H = HIDDEN_DIMENSIONALITY
"""

import jax
from flax import nnx
import optax
import jax.numpy as jnp
from typing import Any
import chex
import numpy as np
from bts.models import sequences


@chex.dataclass
class OutputDistribution:
    """We model the pitch type distribution as a simple categorical
    distribution, and we model the pitch features distribution as
    a mixture of Gaussians.  This dataclass stores the parametrs of
    these distributions.
    """

    # for predicting pitch type
    logits: list[jax.Array]  # (B, S, V)

    # for predicting pitch features
    mixture_weights: jax.Array  # (B, S, M)
    mixture_means: jax.Array  # (B, S, M, F)
    mixture_cov: jax.Array  # (B, S, M, F, F)

    def loss_fn(
        self,
        batch: sequences.PitchInfoBlock,
    ) -> tuple[float, Any]:
        masks = jnp.moveaxis(batch.categorical_missing_mask, -1, 0)
        outcomes = jnp.moveaxis(batch.categorical, -1, 0)
        categorical_out = [
            masked_crossentropy(logits, outcomes, mask)
            for logits, outcomes, mask in zip(self.logits, outcomes, masks)
        ]
        cat_loss = sum(x[0] for x in categorical_out)
        cat_per_dim_losses = jnp.array([x[1] for x in categorical_out])

        real_loss, real_per_dim_losses = batched_gmm_log_loss_with_metrics(
            weights=self.mixture_weights,
            mu=self.mixture_means,
            cov=self.mixture_cov,
            data=batch.numerical,
            missing_mask=batch.numerical_missing_mask,
        )
        # TODO: think about how to normalize across categorical / real losses
        loss = cat_loss + real_loss
        aux = cat_per_dim_losses, real_per_dim_losses
        return loss, aux


def masked_crossentropy(
    logits: jax.Array,  # shape (B, S, V)
    labels: jax.Array,  # shape (B, S)
    missing_mask: jax.Array  # shape (B, S)
) -> jax.Array:
    """Calculates masked cross-entropy loss for sequence data.

    This function computes the cross-entropy loss between predicted logits and true
    labels, but it masks out the loss for timesteps where the label is considered
    "missing" or irrelevant, as indicated by the `missing_mask`. This is commonly
    used in sequence-to-sequence models where input sequences might have variable
    lengths or padded regions.

    Args:
        logits: Predicted logits from the model.
        labels: True labels (integer class indices).
        missing_mask: Binary mask indicating valid (1) and masked (0) timesteps.

    Returns:
        jax.Array: The masked cross-entropy loss, averaged over the valid timesteps
            in the batch. Returns a scalar value.
    """
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    )
    loss = (losses * missing_mask).sum()
    aux = loss / missing_mask.sum()
    return loss, aux



def clean(
    x: jax.Array, # (F,)
    mu: jax.Array, # (M, F)
    cov: jax.Array, # (M, F, F)
    missing_mask: jax.Array # (F,)
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply missing_mask to data points and Gaussian mixture parameters.

    This ensures that hte contribution from entries of x that are missing
    are zeroed out in subsequent loss calculations.
    """
    x = x * missing_mask
    mu = mu * missing_mask[None]
    default = 0.5 * jnp.eye(x.size) / jnp.pi
    cov = jnp.where(
        jnp.outer(missing_mask, missing_mask)[None],
        cov,
        default
    )
    return x, mu, cov


def gmm_log_likelihood(
    x: jax.Array, # (F,)
    mu: jax.Array, # (M, F)
    cov: jax.Array, # (M, F, F)
    weights: jax.Array, # (M,)
) -> float:
    """Compute the log likelihood of data given Gaussian mixture parameters."""

    def component_logpdfs(mu, cov):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mu, cov)

    logpdfs = jax.vmap(component_logpdfs)(mu, cov)
    return jax.scipy.special.logsumexp(logpdfs, b=weights)

def gmm_dim_i_conditional_log_likelihood(
    x: jax.Array, # (F,)
    mu: jax.Array, # (M, F)
    cov: jax.Array, # (M, F, F)
    weights: jax.Array, # (M,)
    i: int
) -> float:
    """Compute log P(x_i | x_{-i})."""
    
    x = jnp.roll(x, shift=-i)
    mu = jnp.roll(mu, shift=-i, axis=1)
    cov = jnp.roll(cov, shift=-i, axis=(1,2))
    
    x_i = x[0]
    x_o = x[1:]
    
    def component_likelihood(m):
        mu_i = mu[m, 0]
        mu_o = mu[m, 1:]
        
        cov_ii = cov[m, 0, 0]           # scalar
        cov_io = cov[m, 0, 1:]          # (F-1,)
        cov_oi = cov[m, 1:, 0]          # (F-1,)
        cov_oo = cov[m, 1:, 1:]         # (F-1, F-1)
        
        L = jnp.linalg.cholesky(cov_oo)
        diff = x_o - mu_o
        v1 = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        v2 = jax.scipy.linalg.solve_triangular(L, cov_oi, lower=True)
        
        # Compute conditional mean: mu_i + cov_io @ inv(cov_oo) @ (x_o - mu_o)
        cond_mean = mu_i + jnp.dot(cov_io, jax.scipy.linalg.solve_triangular(L.T, v1, lower=False))
        # Compute conditional variance: cov_ii - cov_io @ inv(cov_oo) @ cov_oi
        cond_var = cov_ii - jnp.sum(v2 ** 2)

        return jax.scipy.stats.norm.logpdf(x_i, loc=cond_mean, scale=jnp.sqrt(cond_var))

    M = weights.shape[0]
    component_loglikes = jax.vmap(component_likelihood)(jnp.arange(M))
    return jax.scipy.special.logsumexp(component_loglikes, b=weights)


def gmm_per_dim_conditional_log_likelihood(
    x: jax.Array, # (F,)
    mu: jax.Array, # (M, F)
    cov: jax.Array, # (M, F, F)
    weights: jax.Array, # (M,)
) -> jax.Array:
    def foo(i):
        return gmm_dim_i_conditional_log_likelihood(x, mu, cov, weights, i)
    if x.size == 0:
        return jnp.array([])
    return jax.vmap(foo)(jnp.arange(x.size))

def gmm_log_likelihood_with_metrics(
    x: jax.Array, # (F,)
    mu: jax.Array, # (M, F)
    cov: jax.Array, # (M, F, F)
    weights: jax.Array, # (M,)
    missing_mask: jax.Array, # (F,)
) -> tuple[float, jax.Array]:
    x, mu, cov = clean(x, mu, cov, missing_mask)
    result = gmm_log_likelihood(x, mu, cov, weights)
    metrics = gmm_per_dim_conditional_log_likelihood(x, mu, cov, weights)

    return result, metrics


def batched_gmm_log_loss_with_metrics(
    *,
    weights: jax.Array, # (B, S, M)
    mu: jax.Array, # (B, S, M, F)
    cov: jax.Array, # (B, S, M, F, F)
    data: jax.Array, # (B, S, F,)
    missing_mask: jax.Array, # (B, S, F)
):
    """Computes the log likelihood of data given Gaussian mixture parameters.

    Also returns the per-dimension conditional log liklihood as an auxialiary output.
    """
    def foo(*args):
        return jax.tree.map(lambda x: x.sum(axis=0), 
                            jax.vmap(gmm_log_likelihood_with_metrics)(*args))

    def bar(*args):
        return jax.tree.map(lambda x: x.sum(axis=0), jax.vmap(foo)(*args))

    out, metrics = bar(data, mu, cov, weights, missing_mask)

    denom = missing_mask.sum()
    denom_per_dim = missing_mask.sum(axis=(0,1))
    loss = -out #-jnp.where(denom != 0, out / denom, 0)
    metrics = -jnp.where(denom_per_dim != 0, metrics / denom_per_dim, 0)
    return loss, metrics
