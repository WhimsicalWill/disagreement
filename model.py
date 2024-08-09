import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


class MLP(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(768)(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (-1, 28, 28))  # Reshape to correct size
        return x

model = MLP(hidden_dim=128)

x1 = jnp.empty((1, 28, 28, 1))
params = model.init(jax.random.key(42), x1)