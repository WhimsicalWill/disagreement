from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch


MAX_SAMPLES_PER_CLASS = 100

def load_mnist() -> jnp.ndarray:
    """Loads the MNIST dataset into a structured JAX array.

    Returns:
        A 4D JAX array of shape (10, MAX_SAMPLES_PER_CLASS, 28, 28) where
        the first dimension indexes the digit classes (0-9) and the second dimension indexes
        individual samples.
    """
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize data to interval [-1, 1]
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    
    data_by_class = [[] for _ in range(10)]
    
    # Collect up to MAX_SAMPLES_PER_CLASS samples per class
    for img, label in mnist_train:
        if len(data_by_class[label]) < MAX_SAMPLES_PER_CLASS:
            data_by_class[label].append(jnp.array(img.squeeze()))
            # Break if all classes have reached the max samples
            if all(len(data_by_class[l]) == MAX_SAMPLES_PER_CLASS for l in range(10)):
                break

    # Convert the 2D list of numpy arrays to a 4D JAX array of shape (10, N, 28, 28)
    return jnp.array([jnp.array(images) for images in data_by_class])


class MNISTOneStep(Dataset):
    '''This dataset simulates a one-step MNIST env with the below rules:

    - A 0 digit transitions to another 0 digit
    - A 1 digit transitions to a random digit 2-9

    In shape comments, I use N to represent MAX_SAMPLES_PER_CLASS
    '''

    def __init__(self, shuffle=True):
        self.data_array = load_mnist()  # MNIST data array of shape (10, N, 28, 28)
        self.rng = jax.random.PRNGKey(42)
        self.shuffle = shuffle
        self.pairs = self.prepare_pairs()

    def prepare_pairs(self):
        # Pair 0s with another 0 digit
        zeros = self.data_array[0]
        self.rng, subkey = jax.random.split(self.rng)
        shuffled_zeros = zeros[jax.random.permutation(subkey, zeros.shape[0])]
        zero_pairs = jnp.reshape(shuffled_zeros, (-1, 2, 28, 28))  # (N // 2, 2, 28, 28)
        
        # Pair 1s with a random digit from 2-9
        ones = self.data_array[1][:MAX_SAMPLES_PER_CLASS // 2]  # (N // 2, 28, 28)
        self.rng, subkey = jax.random.split(self.rng)

        # Random digits from 2-9
        paired_digits = jax.random.randint(subkey, (MAX_SAMPLES_PER_CLASS // 2,), 2, 10) # (N // 2,)
        # Random indices within the digit classes
        digit_indices = jax.random.choice(subkey, jnp.arange(MAX_SAMPLES_PER_CLASS), shape=(MAX_SAMPLES_PER_CLASS // 2,), replace=False)  # (N // 2,)

        # Fetch images using vectorized indexing
        def fetch_image(digit, idx):
            return self.data_array[digit][idx]
        
        fetch_image_vmap = jax.vmap(fetch_image, in_axes=(0, 0))
        paired_images = fetch_image_vmap(paired_digits, digit_indices)  # (N // 2, 28, 28)
        one_pairs = jnp.stack([ones, paired_images], axis=1)

        # Concatenate all pairs
        pairs = jnp.concatenate([zero_pairs, one_pairs], axis=0)  # (N, 2, 28, 28)
        if self.shuffle:
            self.rng, subkey = jax.random.split(self.rng)
            pairs = pairs[jax.random.permutation(subkey, pairs.shape[0])]
        return pairs

    def __len__(self):
        return self.pairs.shape[0]
    
    def __getitem__(self, idx):
        return self.pairs[idx]
