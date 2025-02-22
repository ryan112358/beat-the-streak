import unittest
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from bts.models import losses  # Assuming losses.py is in the bts/models directory
from bts.models.transformer import TransformerBlock, Transformer, train_step  # Import the classes and function to test


class TransformerBlockTest(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 32
        self.num_heads = 4
        self.rngs = nnx.Rngs(0)
        self.dropout_rate = 0.1
        self.dtype = jnp.float32
        self.batch_size = 2
        self.seq_len = 10

        self.transformer_block = TransformerBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            rngs=self.rngs,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )
        self.input_array = jnp.ones((self.batch_size, self.seq_len, self.hidden_dim), dtype=self.dtype)
        self.mask = nnx.make_causal_mask(self.input_array[:,:,0])

    def test_initialization(self):
        self.assertIsInstance(self.transformer_block.attention, nnx.MultiHeadAttention)
        self.assertIsInstance(self.transformer_block.layernorm, nnx.LayerNorm)
        self.assertIsInstance(self.transformer_block.linear1, nnx.Linear)
        self.assertIsInstance(self.transformer_block.linear2, nnx.Linear)
        self.assertIsInstance(self.transformer_block.dropout, nnx.Dropout)

    def test_forward_pass_no_mask_deterministic(self):
        output = self.transformer_block(self.input_array, deterministic=True)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(output.dtype, self.dtype)

    def test_forward_pass_with_mask_deterministic(self):
        output = self.transformer_block(self.input_array, mask=self.mask, deterministic=True)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(output.dtype, self.dtype)

    def test_forward_pass_no_mask_dropout(self):
        output = self.transformer_block(self.input_array, deterministic=False)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(output.dtype, self.dtype)

    def test_forward_pass_with_mask_dropout(self):
        output = self.transformer_block(self.input_array, mask=self.mask, deterministic=False)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(output.dtype, self.dtype)


class TransformerTest(unittest.TestCase):
    def setUp(self):
        self.seq_len = 10
        self.num_layers = 2
        self.hidden_dim = 32
        self.num_heads = 4
        self.vocab_size = 5
        self.num_numerical_features = 2
        self.mixture_components = 3
        self.dropout_rate = 0.1
        self.dtype = jnp.float32
        self.seed = 0
        self.batch_size = 2

        self.transformer = Transformer(
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            num_numerical_features=self.num_numerical_features,
            mixture_components=self.mixture_components,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            seed=self.seed,
        )
        self.pitch_types = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
        self.pitch_locs = jnp.ones((self.batch_size, self.seq_len, self.num_numerical_features), dtype=self.dtype)
        self.mask = nnx.make_causal_mask(self.pitch_types)

    def test_initialization(self):
        self.assertIsInstance(self.transformer.type_embed, nnx.Embed)
        self.assertIsInstance(self.transformer.loc_embed, nnx.Linear)
        self.assertIsInstance(self.transformer.position_embed, nnx.Embed)
        self.assertEqual(len(self.transformer.blocks), self.num_layers)
        for block in self.transformer.blocks:
            self.assertIsInstance(block, TransformerBlock)
        self.assertIsInstance(self.transformer.type_final_logits, nnx.Linear)
        self.assertIsInstance(self.transformer.numeric_final_weights, nnx.Linear)
        self.assertIsInstance(self.transformer.numeric_final_means, nnx.Linear)
        self.assertIsInstance(self.transformer.numeric_final_stddevs, nnx.Linear)

    def test_forward_pass_no_mask_deterministic(self):
        logits, weights, means, stddevs = self.transformer(
            self.pitch_types, self.pitch_locs, deterministic=True
        )
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.mixture_components))
        self.assertEqual(means.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(stddevs.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(logits.dtype, self.dtype)
        self.assertEqual(weights.dtype, self.dtype)
        self.assertEqual(means.dtype, self.dtype)
        self.assertEqual(stddevs.dtype, self.dtype)

    def test_forward_pass_with_mask_deterministic(self):
        logits, weights, means, stddevs = self.transformer(
            self.pitch_types, self.pitch_locs, mask=self.mask, deterministic=True
        )
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.mixture_components))
        self.assertEqual(means.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(stddevs.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(logits.dtype, self.dtype)
        self.assertEqual(weights.dtype, self.dtype)
        self.assertEqual(means.dtype, self.dtype)
        self.assertEqual(stddevs.dtype, self.dtype)

    def test_forward_pass_no_mask_dropout(self):
        logits, weights, means, stddevs = self.transformer(
            self.pitch_types, self.pitch_locs, deterministic=False
        )
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.mixture_components))
        self.assertEqual(means.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(stddevs.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(logits.dtype, self.dtype)
        self.assertEqual(weights.dtype, self.dtype)
        self.assertEqual(means.dtype, self.dtype)
        self.assertEqual(stddevs.dtype, self.dtype)

    def test_forward_pass_with_mask_dropout(self):
        logits, weights, means, stddevs = self.transformer(
            self.pitch_types, self.pitch_locs, mask=self.mask, deterministic=False
        )
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.mixture_components))
        self.assertEqual(means.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(stddevs.shape, (
            self.batch_size, self.seq_len, self.mixture_components * self.num_numerical_features
        ))
        self.assertEqual(logits.dtype, self.dtype)
        self.assertEqual(weights.dtype, self.dtype)
        self.assertEqual(means.dtype, self.dtype)
        self.assertEqual(stddevs.dtype, self.dtype)


class TrainStepTest(unittest.TestCase):
    def setUp(self):
        self.seq_len = 10
        self.num_layers = 2
        self.hidden_dim = 32
        self.num_heads = 4
        self.vocab_size = 5
        self.num_numerical_features = 2
        self.mixture_components = 3
        self.dropout_rate = 0.1
        self.dtype = jnp.float32
        self.seed = 0
        self.batch_size = 2
        self.learning_rate = 1e-3

        self.model = Transformer(
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            num_numerical_features=self.num_numerical_features,
            mixture_components=self.mixture_components,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            seed=self.seed,
        )
        self.optimizer_def = optax.adam(self.learning_rate)
        self.optimizer = nnx.Optimizer(self.model, self.optimizer_def)

        self.ptypes = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
        self.plocs = jnp.ones((self.batch_size, self.seq_len, self.num_numerical_features), dtype=self.dtype)
        self.type_missing_mask = jnp.ones((self.batch_size, self.seq_len), dtype=self.dtype)
        self.loc_missing_mask = jnp.ones((self.batch_size, self.seq_len, self.num_numerical_features), dtype=self.dtype)

    def test_train_step_runs_without_error(self):
        aux = train_step(
            self.model,
            self.optimizer,
            self.ptypes,
            self.plocs,
            self.type_missing_mask,
            self.loc_missing_mask,
        )
        self.assertIsInstance(aux, tuple)
        self.assertEqual(len(aux), 2)
        type_loss, real_loss = aux
        self.assertIsInstance(type_loss, jax.Array)
        self.assertIsInstance(real_loss, jax.Array)


if __name__ == '__main__':
    unittest.main()
