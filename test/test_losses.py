import unittest
import jax
import jax.numpy as jnp
import jax.scipy.stats
import optax
import numpy as np

from bts.models.losses import (
    _logpdf_with_missing,
    masked_crossentropy,
    _single_mixture_density_loss,
    mixture_density_loss,
)


class TestLossFunctions(unittest.TestCase):

    def test_logpdf_with_missing_no_missing(self):
        key = jax.random.PRNGKey(0)
        features = 3
        x = jax.random.normal(key, (features,))
        mu = jax.random.normal(key, (features,))
        cov = jnp.eye(features)
        missing_mask = jnp.ones_like(x)

        expected_logpdf = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mu, cov=cov
        )
        actual_logpdf = _logpdf_with_missing(x, mu, cov, missing_mask)
        self.assertTrue(jnp.allclose(actual_logpdf, expected_logpdf))

    def test_logpdf_with_missing_some_missing(self):
        key = jax.random.PRNGKey(0)
        features = 3
        x = jnp.array([1.0, 2.0, 3.0])
        mu = jnp.array([0.0, 0.0, 0.0])
        cov = jnp.eye(features)
        missing_mask = jnp.array([1.0, 0.0, 1.0])

        x_observed = x[missing_mask == 1]
        mu_observed = mu[missing_mask == 1]
        cov_observed = cov[missing_mask == 1, :][:, missing_mask == 1]
        expected_logpdf = jax.scipy.stats.multivariate_normal.logpdf(
            x_observed, mean=mu_observed, cov=cov_observed
        )
        actual_logpdf = _logpdf_with_missing(x, mu, cov, missing_mask)

        self.assertTrue(jnp.allclose(actual_logpdf, expected_logpdf))

    def test_logpdf_with_missing_all_missing(self):
        key = jax.random.PRNGKey(0)
        features = 3
        x = jax.random.normal(key, (features,))
        mu = jax.random.normal(key, (features,))
        cov = jnp.eye(features)
        missing_mask = jnp.zeros_like(x)

        actual_logpdf = _logpdf_with_missing(x, mu, cov, missing_mask)

        np.testing.assert_allclose(actual_logpdf, 0.0, atol=1e-6)

    def test_masked_crossentropy_no_masking(self):
        key = jax.random.PRNGKey(0)
        batch_size = 2
        seq_len = 3
        vocab_size = 5
        logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        labels = jnp.array([[1, 2, 0], [3, 4, 1]])
        missing_mask = jnp.ones((batch_size, seq_len))

        expected_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        actual_loss = masked_crossentropy(logits, labels, missing_mask)
        self.assertTrue(jnp.allclose(actual_loss, expected_loss))

    def test_masked_crossentropy_with_masking(self):
        key = jax.random.PRNGKey(0)
        batch_size = 2
        seq_len = 3
        vocab_size = 5
        logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        labels = jnp.array([[1, 2, 0], [3, 4, 1]])
        missing_mask = jnp.array([[1, 1, 0], [1, 0, 1]])  # Masking some tokens

        losses = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        masked_losses = losses * missing_mask
        expected_loss = masked_losses.sum() / missing_mask.sum()
        actual_loss = masked_crossentropy(logits, labels, missing_mask)
        self.assertTrue(jnp.allclose(actual_loss, expected_loss))

    def test_masked_crossentropy_all_masked(self):
        key = jax.random.PRNGKey(0)
        batch_size = 2
        seq_len = 3
        vocab_size = 5
        logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        labels = jnp.array([[1, 2, 0], [3, 4, 1]])
        missing_mask = jnp.zeros((batch_size, seq_len))

        actual_loss = masked_crossentropy(logits, labels, missing_mask)
        self.assertEqual(actual_loss, 0.0)

    def test_single_mixture_density_loss_single_component(self):
        key = jax.random.PRNGKey(0)
        features = 2
        pitches = jnp.array([1.0, 2.0])
        missing_mask = jnp.ones_like(pitches)
        weights = jnp.array([1.0])  # Single component, weight effectively 1
        means = jnp.array([0.0, 0.0])
        variances = jnp.array([1.0, 1.0])

        expected_loss = -_logpdf_with_missing(
            pitches, means, jnp.diag(variances.reshape(1, -1)[0]), missing_mask
        )  # Negative logpdf
        actual_loss = _single_mixture_density_loss(
            weights, means, variances, pitches, missing_mask
        )
        self.assertTrue(jnp.allclose(actual_loss, expected_loss))

    def test_single_mixture_density_loss_multiple_components(self):
        key = jax.random.PRNGKey(0)
        features = 2
        mixture_components = 3
        pitches = jnp.array([1.0, 2.0])
        missing_mask = jnp.ones_like(pitches)
        weights = jnp.array([0.3, 0.5, 0.2])
        means = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])  # 3 components * 2 features
        variances = jnp.array(
            [1.0, 1.0, 0.5, 0.5, 0.2, 0.2]
        )  # 3 components * 2 features

        means_reshaped = means.reshape(mixture_components, features)
        variances_reshaped = variances.reshape(mixture_components, features)

        logpdfs = jnp.array(
            [
                _logpdf_with_missing(
                    pitches,
                    means_reshaped[i],
                    jnp.diag(variances_reshaped[i]),
                    missing_mask,
                )
                for i in range(mixture_components)
            ]
        )
        expected_loss = -jax.scipy.special.logsumexp(logpdfs, b=weights)
        actual_loss = _single_mixture_density_loss(
            weights, means, variances, pitches, missing_mask
        )

        self.assertTrue(jnp.allclose(actual_loss, expected_loss))

    def test_single_mixture_density_loss_missing_pitches(self):
        key = jax.random.PRNGKey(0)
        features = 2
        mixture_components = 2
        pitches = jnp.array([1.0, jnp.nan])  # Second pitch is missing (will be masked)
        missing_mask = jnp.array([1.0, 0.0])
        weights = jnp.array([0.5, 0.5])
        means = jnp.array([0.0, 0.0, 1.0, 1.0])
        variances = jnp.array([1.0, 1.0, 0.5, 0.5])

        means_reshaped = means.reshape(mixture_components, features)
        variances_reshaped = variances.reshape(mixture_components, features)

        logpdfs = jnp.array(
            [
                _logpdf_with_missing(
                    pitches,
                    means_reshaped[i],
                    jnp.diag(variances_reshaped[i]),
                    missing_mask,
                )
                for i in range(mixture_components)
            ]
        )
        expected_loss = -jax.scipy.special.logsumexp(logpdfs, b=weights)
        actual_loss = _single_mixture_density_loss(
            weights, means, variances, pitches, missing_mask
        )

        np.testing.assert_allclose(actual_loss, expected_loss)

    def test_mixture_density_loss_single_sequence(self):
        key = jax.random.PRNGKey(0)
        batch_size = 1
        seq_len = 2
        features = 2
        mixture_components = 2
        pitches = jax.random.normal(key, (batch_size, seq_len, features))
        missing_masks = jnp.ones((batch_size, seq_len, features))
        weights = jax.random.uniform(key, (batch_size, seq_len, mixture_components))
        means = jax.random.normal(
            key, (batch_size, seq_len, mixture_components * features)
        )
        variances = jnp.ones((batch_size, seq_len, mixture_components * features))

        # Calculate per-token losses manually for comparison
        per_token_losses_manual = jnp.zeros((batch_size, seq_len))
        for b in range(batch_size):
            for s in range(seq_len):
                per_token_losses_manual = per_token_losses_manual.at[b, s].set(
                    _single_mixture_density_loss(
                        weights[b, s],
                        means[b, s],
                        variances[b, s],
                        pitches[b, s],
                        missing_masks[b, s],
                    )
                )
        expected_loss = per_token_losses_manual.mean()
        actual_loss = mixture_density_loss(
            weights, means, variances, pitches, missing_masks
        )
        self.assertTrue(jnp.allclose(actual_loss, expected_loss))

    def test_mixture_density_loss_batch_sequences(self):
        key = jax.random.PRNGKey(0)
        batch_size = 2
        seq_len = 3
        features = 2
        mixture_components = 2
        pitches = jax.random.normal(key, (batch_size, seq_len, features))
        missing_masks = jnp.ones((batch_size, seq_len, features))
        weights = jax.random.uniform(key, (batch_size, seq_len, mixture_components))
        means = jax.random.normal(
            key, (batch_size, seq_len, mixture_components * features)
        )
        variances = jnp.ones((batch_size, seq_len, mixture_components * features))

        # Calculate per-token losses manually for comparison
        per_token_losses_manual = jnp.zeros((batch_size, seq_len))
        for b in range(batch_size):
            for s in range(seq_len):
                per_token_losses_manual = per_token_losses_manual.at[b, s].set(
                    _single_mixture_density_loss(
                        weights[b, s],
                        means[b, s],
                        variances[b, s],
                        pitches[b, s],
                        missing_masks[b, s],
                    )
                )
        expected_loss = per_token_losses_manual.mean()
        actual_loss = mixture_density_loss(
            weights, means, variances, pitches, missing_masks
        )
        self.assertTrue(jnp.allclose(actual_loss, expected_loss))

    def test_mixture_density_loss_masked_sequences(self):
        key = jax.random.PRNGKey(0)
        batch_size = 2
        seq_len = 3
        features = 2
        mixture_components = 2
        pitches = jax.random.normal(key, (batch_size, seq_len, features))
        missing_masks = jnp.array(
            [[[1, 1], [1, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]]
        )  # Masking some features and tokens
        weights = jax.random.uniform(key, (batch_size, seq_len, mixture_components))
        means = jax.random.normal(
            key, (batch_size, seq_len, mixture_components * features)
        )
        variances = jnp.ones((batch_size, seq_len, mixture_components * features))

        # Calculate per-token losses manually for comparison
        per_token_losses_manual = jnp.zeros((batch_size, seq_len))
        for b in range(batch_size):
            for s in range(seq_len):
                per_token_losses_manual = per_token_losses_manual.at[b, s].set(
                    _single_mixture_density_loss(
                        weights[b, s],
                        means[b, s],
                        variances[b, s],
                        pitches[b, s],
                        missing_masks[b, s],
                    )
                )
        expected_loss = per_token_losses_manual.mean()

        actual_loss = mixture_density_loss(
            weights, means, variances, pitches, missing_masks
        )
        self.assertTrue(jnp.allclose(actual_loss, expected_loss))


if __name__ == "__main__":
    unittest.main()
