import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from MovingMNIST.MovingMNIST import MovingMNIST
import jax.numpy as jnp

def load_moving_mnist(config):
	transform = transforms.Compose([
		transforms.ToTensor(),  # Automatically converts to [0, 1] range
	])

	train_dataset = MovingMNIST(root='./data/mnist', train=True, download=True, transform=transform)
	test_dataset = MovingMNIST(root='./data/mnist', train=False, download=True, transform=transform)

	def numpy_collate(batch):
		# Both videos_start and videos_end are collated to have shapes (B, 10, 64, 64, 1)
		videos_start, videos_end = zip(*batch)
		videos_start = jnp.array([jnp.expand_dims(video.numpy(), axis=-1) for video in videos_start])
		videos_end = jnp.array([jnp.expand_dims(video.numpy(), axis=-1) for video in videos_end])
		return videos_start, videos_end

	train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=numpy_collate)
	test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=numpy_collate)

	return train_loader, test_loader