import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from jax import random, nn, grad, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax
from tqdm import tqdm
from functools import partial


# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def numpy_collate(batch):
    images, labels = zip(*batch)
    images = jnp.array([image.numpy() for image in images])
    labels = jnp.array(labels)
    return images, labels

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=numpy_collate)

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes) - 1)
    params = [{'w': random.normal(k, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / (sizes[i] + sizes[i + 1])),
               'b': jnp.zeros((sizes[i + 1],))}
              for i, k in enumerate(keys)]
    return params

def relu(x):
    return jnp.maximum(0, x)

@partial(vmap, in_axes=(None, 0))
def predict(params, image):
    activations = image.reshape(-1)
    for param in params[:-1]:
        activations = relu(jnp.dot(activations, param['w']) + param['b'])
    final_layer = params[-1]
    logits = jnp.dot(activations, final_layer['w']) + final_layer['b']
    return logits - logsumexp(logits)

@jit
def loss(params, images, labels):
    preds = predict(params, images)
    return -jnp.mean(jnp.sum(preds * labels, axis=1))

@jit
def accuracy(params, images, labels):
    preds = predict(params, images)
    return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(labels, axis=1))

# Initialize parameters
N_EPOCHS = 10
layer_sizes = [784, 512, 512, 10]
key = random.PRNGKey(0)
params = init_network_params(layer_sizes, key)

# Define optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

@jit
def update(params, opt_state, images, labels):
    grads = grad(loss)(params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    acc = accuracy(params, images, labels)
    params = optax.apply_updates(params, updates)
    return params, opt_state, acc

for epoch in range(N_EPOCHS):
    train_accs = []
    for batch_images, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        batch_labels = nn.one_hot(batch_labels, 10)
        params, opt_state, acc = update(params, opt_state, batch_images, batch_labels)
        train_accs.append(acc)

    test_accs = []
    for batch_images, batch_labels in tqdm(test_loader, desc=f"Epoch {epoch + 1} Testing"):
        batch_labels = nn.one_hot(batch_labels, 10)
        # TODO: use an annotation for vmap since we don't care about non-batch usage
        acc = accuracy(params, batch_images, batch_labels)
        test_accs.append(acc)

    train_acc = jnp.mean(jnp.array(train_accs))
    test_acc = jnp.mean(jnp.array(test_accs))
    print(f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")