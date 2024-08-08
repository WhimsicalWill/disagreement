from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


@struct.dataclass
class EnvState(environment.EnvState):
    label: int
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: float = -1.0
    max_action: float = 1.0
    img_shape: Tuple[int, int] = (28, 28)
    max_steps_in_episode: int = 1


class MNISTOneStep(environment.Environment[EnvState, EnvParams]):
    """This is a simple 1-step env and has the following properties:

    - On reset, the first digit is either 0 or 1
    - On the next step, 0 digit -> 0 digit and 1 digit -> any digit
    """

    def __init__(self, fraction: float = 1.0):
        super().__init__()
        self.data_array = load_mnist()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition based on MNIST transition rules."""
        next_label = lax.select(
            jnp.equal(state.label, 0),
            0,
            jax.random.choice(key, jnp.arange(2, 10))
        )
        idx = jax.random.randint(key, (), 0, MAX_SAMPLES_PER_CLASS)
        next_obs = self.data_array[next_label][idx]
        next_state = EnvState(label=next_label, time=state.time + 1)
        reward = 0.0  # No rewards since we will use an external intrinsic reward
        done = self.is_terminal(next_state, params)
        info = {"discount": 1.0}
        return lax.stop_gradient(next_obs), lax.stop_gradient(next_state), reward, done, info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by randomly choosing an image labeled 0 or 1."""
        start_label = jax.random.choice(key, jnp.array([0, 1]))
        idx = jax.random.randint(key, (), 0, MAX_SAMPLES_PER_CLASS)
        img = self.data_array[start_label][idx]
        state = EnvState(
            label=start_label,
            time=0,
        )
        return img, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Every step transition is terminal! No long term credit assignment!
        return jnp.array(True)

    @property
    def name(self) -> str:
        """Environment name."""
        return "MNISTOneStep"

    # This shouldn't be used since the env is continuous, but we just return 1 anyway
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams]) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=params.img_shape
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=params.img_shape
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "label": spaces.Discrete(10),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )