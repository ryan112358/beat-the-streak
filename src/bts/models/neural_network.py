import jax
import jax.numpy as jnp
import flax
import optax
import functools
from flax.training import train_state
import flax.linen as nn

class FullyConnectedNeuralNetwork(nn.Module):
    out_dims: int = 1
    hidden_dims: int = 80
    dropout: float = 0.2
    layers: int = 2

    @nn.compact
    def __call__(self, x, train=True):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.layers):
            x = nn.Dense(self.hidden_dims)(x)
            x = nn.Dropout(self.dropout, deterministic=not train)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dims)(x)
        return x

def mean_squared_error(predictions, targets):
    return optax.squared_error(predictions, targets).mean()

def binary_crossentropy(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


@jax.jit
def apply_model(state, X, y, rng=None, regularization=0.003):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    pred = state.apply_fn({'params': params}, X, train=rng != None, rngs=rng).flatten()
    loss = mean_squared_error(pred, y)
    penalty = regularization*optax.global_norm(params)**2
    return loss+penalty, pred

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, pred), grads = grad_fn(state.params)

  top1000 = jnp.argsort(pred)[-1000:]
  mean_hits = y[top1000].mean()
  accuracy = (y[top1000] >= 1).mean()

  return grads, loss, mean_hits, accuracy, pred

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def create_train_state(model, rng, X, learning_rate=0.00007):
  """Creates initial `TrainState`."""
  params = model.init(rng, X, train=False)['params']
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

