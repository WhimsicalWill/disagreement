import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax


class CustomMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

# Example usage:
transform = transforms.Compose([transforms.ToTensor()])
custom_dataset = CustomMNISTDataset(root='data', train=True, transform=transform)


# Define the ensemble size
ENSEMBLE_SIZE = 5

# Initialize the ensemble of models
ensemble_models = [MLP(hidden_dim=128, out_dim=28*28) for _ in range(ENSEMBLE_SIZE)]
ensemble_params = [model.init(jax.random.PRNGKey(i), x1) for i, model in enumerate(ensemble_models)]

def compute_nll_loss(predicted, target):
    return 0.5 * jnp.mean((predicted - target) ** 2)

def compute_intrinsic_reward(predictions):
    means = jnp.stack(predictions, axis=0)
    variance = jnp.var(means, axis=0)
    intrinsic_reward = jnp.mean(variance)
    return intrinsic_reward

# Initialize optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Create a training state for each model
train_states = [train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer) 
                for model, params in zip(ensemble_models, ensemble_params)]

def train_step(state, batch):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['image'])
        loss = compute_nll_loss(predictions, batch['next_image'])
        return loss, predictions

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, predictions

# Example training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        # Train each model in the ensemble
        predictions = []
        for i in range(ENSEMBLE_SIZE):
            train_states[i], loss, prediction = train_step(train_states[i], batch)
            predictions.append(prediction)

        # Compute intrinsic reward
        intrinsic_reward = compute_intrinsic_reward(predictions)
        print(f"Epoch {epoch}, Intrinsic Reward: {intrinsic_reward}")