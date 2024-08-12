import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DigitToDigit
from data import MNISTOneStep
from utils import plot_metrics, plot_predictions


def create_train_state(rng_key, learning_rate):
    model = DigitToDigit(hidden_dim=128)
    params = model.init(rng_key, jnp.ones([1, 28, 28]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def mse_loss(params, apply_fn, inputs, targets):
    preds = apply_fn({'params': params}, inputs)
    return jnp.mean((preds - targets) ** 2)

def train_step(ensemble_states, inputs, targets, metrics):
    '''Runs a train step for each bootstrap in the ensemble on a given minibatch.'''
    for i, state in enumerate(ensemble_states):
        loss_fn = lambda params: mse_loss(params, state.apply_fn, inputs, targets)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
        metrics[f"loss_b{i+1}"].append(loss)
        metrics[f"grad_norms_b{i+1}"].append(grad_norm)
        ensemble_states[i] = state.apply_gradients(grads=grads)
    return ensemble_states, metrics

def intrinsic_reward(ensemble_states, val_zeros, val_ones, metrics):
    '''Computes the intrinsic reward using the ensemble on the validation dataset.
    The reward is computed separately for the 0s and 1s.
    The goal of this experiment is to show that this function is agnostic to
    aleatoric uncertainty in the long run.
    We do this by comparing the uncertainty in the 1 prediction (more entropy)
    to the uncertainty in the 0 prediction (less entropy)
    '''
    # compute the variance in predictions on the 0 digit
    # update metrics["ir_0"]
    batch_size = val_zeros.shape[0]
    preds_zeros = jnp.zeros((len(ensemble_states), batch_size, 28, 28))
    preds_ones = jnp.zeros((len(ensemble_states), batch_size, 28, 28))
    for i, state in enumerate(ensemble_states):
        preds_zeros = preds_zeros.at[i].set(state.apply_fn({'params': state.params}, val_zeros))
        preds_ones = preds_ones.at[i].set(state.apply_fn({'params': state.params}, val_ones))
    
    # We take the variance across the bootstrap dimension, and then
    # average it across the batch and the features in each prediction
    ir_0 = jnp.mean(jnp.var(preds_zeros, axis=0))
    ir_1 = jnp.mean(jnp.var(preds_ones, axis=0))
    metrics["ir_0"].append(ir_0)
    metrics["ir_1"].append(ir_1)
    return metrics

def train():
    dataset = MNISTOneStep()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training Loop
    num_epochs = 3
    lr = 1e-3
    num_bootstraps = 5

    # Initialize ensemble of bootstraps
    ensemble_states = [create_train_state(jax.random.PRNGKey(i), lr) for i in range(num_bootstraps)]
    
    # Use the first batch of the dataset for validation purposes
    val_zeros, val_ones = dataset.validation_batch

    # Loss and grad norms are tracked for each bootstrap separately
    metrics = {
        "ir_0": [],  # intrinsic rewards for digit 0 validation dataset
        "ir_1": [],  # intrinsic rewards for digit 1 validation dataset
    }
    for i in range(num_bootstraps):
        metrics[f"loss_b{i+1}"] = []
        metrics[f"grad_norms_b{i+1}"] = []

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        for batch in tqdm(data_loader, desc='Batches', leave=False):
            inputs = jnp.array(batch[:, 0, :, :])
            targets = jnp.array(batch[:, 1, :, :])
            ensemble_states, metrics = train_step(ensemble_states, inputs, targets, metrics)
            metrics = intrinsic_reward(ensemble_states, val_zeros, val_ones, metrics)

    plot_metrics(metrics, num_bootstraps)
    plot_predictions(ensemble_states[i], batch)  # visualize the last batch


if __name__ == "__main__":
    train()