import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DigitToDigit
from data import MNISTOneStep


def create_train_state(rng_key, learning_rate):
    model = DigitToDigit(hidden_dim=128)
    params = model.init(rng_key, jnp.ones([1, 28, 28]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def mse_loss(params, apply_fn, inputs, targets):
    preds = apply_fn({'params': params}, inputs)
    return jnp.mean((preds - targets) ** 2)

def train_step(state, inputs, targets):
    # TODO: change this to value and grad so that we can track the loss and its gradient over time
    # We will use matplotlib to plot these quantities over the training steps
    grads = jax.grad(mse_loss, argnums=0)(state.params, state.apply_fn, inputs, targets)
    return state.apply_gradients(grads=grads)

def intrinsic_reward(ensemble, inputs):
    predictions = [state.apply_fn({'params': state.params}, inputs) for state in ensemble]
    variance = jnp.var(jnp.stack(predictions), axis=0)
    return jnp.mean(variance)


dataset = MNISTOneStep()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
num_epochs = 10
lr = 1e-3
ensemble_size = 5

# Initialize ensemble of bootstraps
ensemble = [create_train_state(jax.random.PRNGKey(i), lr) for i in range(ensemble_size)]

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    for batch in tqdm(data_loader, desc='Batches', leave=False):
        inputs = jnp.array(batch[:, 0, :, :])
        targets = jnp.array(batch[:, 1, :, :])
        for i, train_state in enumerate(ensemble):
            ensemble[i] = train_step(train_state, inputs, targets)

    # Compute intrinsic reward
    sample_inputs = inputs  # Using the first sample of the last batch as an example
    reward = intrinsic_reward(ensemble, sample_inputs)
    print(f'Epoch {epoch+1}, Intrinsic Reward: {reward}')
