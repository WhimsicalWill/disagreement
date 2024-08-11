import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


class DigitToDigit(nn.Module):
    hidden_dim: int
    output_dim: int = 784  # 28*28 for reshaping purposes

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        x = nn.tanh(x)
        return x.reshape((-1, 28, 28))  # Reshape back to image dimensions
