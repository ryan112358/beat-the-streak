import jax
import jax.numpy as jnp
import jax.scipy as jsp
import flax
import optax
import functools
from flax.training import train_state
import flax.linen as nn

# jax.config.update('jax_disable_jit', True)  # turnoff temporarily for debugging


class FullyConnectedNeuralNetwork(nn.Module):
    out_dims: int = 1
    hidden_dims: int = 80
    dropout: float = 0.2
    layers: int = 2

    @nn.compact
    def __call__(self, features, train=True):
        """ Call the model and compute the activations

        :param features: Adictionary of jax arrays, (leading dimensions is the batch axis)

        :return: The model activations.
        """
        features = jax.tree_util.tree_map(lambda X: X.reshape(X.shape[0], -1), features)
        x = sum(
            nn.Dense(self.hidden_dims)(X) * imputed
            for X, imputed in features.values()
        )

        # now use a standard feed-forward architecture
        for _ in range(self.layers - 1):
            x = nn.Dropout(self.dropout, deterministic=not train)(x)
            x = nn.relu(x)
            x = x + nn.Dense(self.hidden_dims)(x)
        return nn.Dense(self.out_dims)(nn.relu(x))


def model_squared_difference(pytree1, pytree2):
    diff = jax.tree_util.tree_map(lambda x, y: jnp.sum((x - y) ** 2), pytree1, pytree2)
    return sum(jax.tree_util.tree_flatten(diff)[0])


def mean_squared_error(predictions, targets):
    # Can be used for both (ehit - y) and (ehit - prev_ehit)
    return optax.squared_error(predictions, targets).mean()


def binary_crossentropy(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


def beta_regularization(logits, prev_logits, scale=1000):
    # Assumes we are using binary cross entropy
    p = jnp.exp(logits)
    phat = jnp.exp(prev_logits)
    a = phat * scale
    b = (1 - phat) * scale
    return jsp.stats.beta.logpdf(p, a, b).mean()


@jax.jit
def apply_model(
    state, X, y, rng=None, regularization=0.003, reg2=0.0, previous_state=None
):
    """Computes gradients, loss and accuracy for a single batch."""
    batch_size = list(X.values())[0][0].shape[0]

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, X, train=rng != None, rngs=rng
        ).flatten()
        # loss = mean_squared_error(pred, y)
        loss = binary_crossentropy(logits, y)
        penalty = regularization * optax.global_norm(params) ** 2
        # jax.debug.print('CHECKPT, {x}', x=penalty)
        if previous_state != None:
            penalty += reg2 * model_squared_difference(params, previous_state.params)
        return loss + penalty / batch_size, jax.nn.sigmoid(logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)

    return grads, loss, pred


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(model, rng, X, learning_rate=0.00007):
    """Creates initial `TrainState`."""
    params = model.init(rng, X, train=False)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
