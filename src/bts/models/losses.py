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


@chex.dataclass
class InputData:
    # features to be predicted
    pitch_types: jax.Array  # (B, S)
    pitch_features: jax.Array  # (B, S, F)

    # entries of pitch_types/features to ignore (due to padding or missing data)
    type_missing_mask: jax.Array  # (B, S)
    feature_missing_mask: jax.Array  # (B, S, F)

    # to be used as context for each pitch
    pitch_in_atbat: jax.Array | None = None # (B, S)

    def sample(self, batch_size: int) -> 'InputData':
        idx = np.random.choice(self.pitch_types.shape[0], size=batch_size)
        return jax.tree.map(lambda x: x[idx], self)


@chex.dataclass
class OutputDistribution:
    """We model the pitch type distribution as a simple categorical
    distribution, and we model the pitch features distribution as
    a mixture of Gaussians.  This dataclass stores the parametrs of
    these distributions.
    """
    # for predicting pitch type
    logits: jax.Array  # (B, S, V)

    # for predicting pitch features
    mixture_weights: jax.Array  # (B, S, M)
    mixture_means: jax.Array  # (B, S, M*F)
    mixture_variances: jax.Array  # (B, S, M*F)


def loss_fn(prediction: OutputDistribution, batch: InputData) -> tuple[float, Any]:

    ptypes = batch.pitch_types[:, 1:]
    plocs = batch.pitch_features[:, 1:]
    type_missing_mask = batch.type_missing_mask[:, 1:]
    loc_missing_mask = batch.feature_missing_mask[:, 1:]

    logits = prediction.logits[:, :-1]
    weights = prediction.mixture_weights[:, :-1]
    means = prediction.mixture_means[:, :-1]
    variances = prediction.mixture_variances[:, :-1]

    type_loss = masked_crossentropy(logits, ptypes, type_missing_mask)
    real_loss = mixture_density_loss(
        weights, means, variances, plocs, loc_missing_mask
    )
    return 5 * type_loss + real_loss, (type_loss, real_loss)



def _logpdf_with_missing(
    x: jax.Array,
    mu: jax.Array,
    cov: jax.Array,
    missing_mask: jax.Array,
) -> float:
    """Calculates the multivariate normal log probability density function (log PDF)
    while handling missing values.

    This function computes the log PDF of a multivariate normal distribution, but it
    specifically addresses scenarios where some dimensions of the input `x` might be
    missing. Missing dimensions are indicated by a `missing_mask`. The function
    effectively marginalizes out the missing dimensions when calculating the log PDF.

    Args:
        x: Input array, representing a single data point. Shape should be (FEATURES,).
        mu: Mean vector of the multivariate normal distribution. Shape should be (FEATURES,).
        cov: Covariance matrix of the multivariate normal distribution. Shape should be (FEATURES, FEATURES).
        missing_mask: Binary mask indicating observed (1) and missing (0) dimensions in `x`.
            Shape should be (FEATURES,).

    Returns:
        float: The log PDF value for the given data point `x` under the specified
            multivariate normal distribution, considering the missing dimensions.
    """
    x_clean = x * missing_mask
    mu_clean = mu * missing_mask
    mask2d = jnp.outer(missing_mask, missing_mask)
    default = jnp.diag(
        jnp.ones_like(mu) / (2 * jnp.pi)
    )  # Default covariance for missing dims
    cov_clean = jnp.where(
        mask2d, cov, default
    )  # if both dims are present in mask, use cov, else default

    logpdf_clean = jax.scipy.stats.multivariate_normal.logpdf(
        x_clean, mean=mu_clean, cov=cov_clean
    )

    return logpdf_clean


def masked_crossentropy(
    logits: jax.Array, labels: jax.Array, missing_mask: jax.Array
) -> jax.Array:
    """Calculates masked cross-entropy loss for sequence data.

    This function computes the cross-entropy loss between predicted logits and true
    labels, but it masks out the loss for timesteps where the label is considered
    "missing" or irrelevant, as indicated by the `missing_mask`. This is commonly
    used in sequence-to-sequence models where input sequences might have variable
    lengths or padded regions.

    Args:
        logits: Predicted logits from the model.
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE).
        labels: True labels (integer class indices).
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH).
        missing_mask: Binary mask indicating valid (1) and masked (0) timesteps.
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH).

    Returns:
        jax.Array: The masked cross-entropy loss, averaged over the valid timesteps
            in the batch. Returns a scalar value.
    """
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    )
    loss = (losses * missing_mask).sum() / missing_mask.sum()
    return jnp.nan_to_num(loss)


def _single_mixture_density_loss(
    weights: jax.Array,
    means: jax.Array, 
    variances: jax.Array,
    pitches: jax.Array, 
    missing_mask: jax.Array,
):
    """Calculates the negative log-likelihood loss for a single data point
    under a Mixture Density Network (MDN) model, handling missing values.

    This function computes the loss for a single data point (e.g., a single timestep
    in a sequence) given the parameters of an MDN. The MDN is assumed to output
    mixture weights, means, and variances for a Gaussian mixture model. The loss is
    the negative log-likelihood of the data point under this mixture distribution.
    It also incorporates a `missing_mask` to handle potentially missing features
    in the data point.

    Args:
        weights: Mixture weights for each component in the MDN.
            Shape should be (MIXTURE_COMPONENTS,).
        means: Means of the Gaussian components in the MDN.
            Shape should be (MIXTURE_COMPONENTS * FEATURES,).
        variances: Variances of the Gaussian components in the MDN.
            Shape should be (MIXTURE_COMPONENTS * FEATURES,).
        pitches: The target data point (e.g., pitch values).
            Shape should be (FEATURES,).
        missing_mask: Binary mask indicating observed (1) and missing (0) features in `pitches`.
            Shape should be (FEATURES,).

    Returns:
        float: The negative log-likelihood loss for the single data point under the MDN.
    """
    mixture_components = weights.shape[0]
    means = means.reshape(mixture_components, -1)
    variances = variances.reshape(mixture_components, -1)

    def single_logpdf(mu, var):
        return _logpdf_with_missing(pitches, mu, jnp.diag(var), missing_mask)
        # return jax.scipy.stats.multivariate_normal.logpdf(pitches, mu, jnp.diag(var))

    logpdfs = jax.vmap(single_logpdf)(means, variances)
    return -jax.scipy.special.logsumexp(logpdfs, axis=0, b=weights)


def mixture_density_loss(weights, means, variances, pitches, missing_masks):
    """Calculates the average negative log-likelihood loss over a batch of sequences
    under a Mixture Density Network (MDN) model, handling missing values.

    This function extends `_single_mixture_density_loss` to handle batches of sequence
    data. It assumes the input arrays have leading dimensions for batch size and
    sequence length (BATCH_SIZE, SEQUENCE_LENGTH, ...). It calculates the MDN loss
    for each timestep in each sequence and then averages the loss over all valid
    timesteps across the batch. Missing values in the input sequences are handled
    using `missing_masks`.

    Args:
        weights: Mixture weights for each component in the MDN.
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH, MIXTURE_COMPONENTS).
        means: Means of the Gaussian components in the MDN.
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH, MIXTURE_COMPONENTS * FEATURES).
        variances: Variances of the Gaussian components in the MDN.
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH, MIXTURE_COMPONENTS * FEATURES).
        pitches: The target sequence data (e.g., pitch values).
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH, FEATURES).
        missing_masks: Binary masks indicating observed (1) and missing (0) features in `pitches`.
            Shape should be (BATCH_SIZE, SEQUENCE_LENGTH, FEATURES).

    Returns:
        jax.Array: The average negative log-likelihood loss over the batch.
            Returns a scalar value.
    """
    # Same as _single_mixture_density_loss with leading (batch_size, seq_len) axes
    per_token_losses = jax.vmap(jax.vmap(_single_mixture_density_loss))(
        weights, means, variances, pitches, missing_masks
    )
    # not sure if this is the best way to normalize, mean seems reasonable for average loss
    return per_token_losses.mean()
    # return (jnp.nan_to_num(per_token_losses) * masks).sum() / masks).sum() 
