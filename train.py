import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
from tqdm import tqdm
from nets import RSSM
from dataset import load_moving_mnist
from itertools import islice
from utils import OneHotDist, make_video_comparison, save_video

sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


@jax.vmap
def mse(preds, targets):
	return optax.squared_error(preds, targets)


def compute_metrics(preds, targets):
	# mse function returns array of losses of shape (B,)
	mse_loss = mse(preds, targets).mean()
	return {'mse': mse_loss}


def train_step(rssm, params, videos, sample_rng, config):
	def loss_fn(params):
		# prepare initial state and use model to predict output
		batch_size = videos.shape[0]
		deter = jnp.zeros((batch_size, config['DETER_DIM']))
		_, outs = rssm.apply({'params': params}, deter, videos, method=rssm.observe, rngs={'sample': sample_rng})
		
		# compute losses
		mse_loss = mse(outs['recon_obs'], videos).mean()
		dyn = OneHotDist(sg(outs['post_logits'])).kl_divergence(OneHotDist(outs['prior_logits']))
		rep = OneHotDist(outs['post_logits']).kl_divergence(OneHotDist(sg(outs['prior_logits'])))
		dyn = jnp.maximum(dyn, config['FREE_BITS']).mean()
		rep = jnp.maximum(rep, config['FREE_BITS']).mean()
		kld_loss = dyn + config['KL_BALANCE'] * rep
		loss = mse_loss + config['BETA'] * kld_loss
		return loss

	grads = jax.grad(loss_fn)(params)
	return grads


def eval_f(rssm, params, test_loader, rng, config):
	'''
	Args:
		rssm: The RSSM model
		params: The params for the model
		test_loader: The DataLoader for the MovingMNIST test dataset
		rng: The rng to be used when computing test metrics for reproducibility
		config: the config
	Returns:
		The metrics and comparison visualization
	'''
	def eval_model(rssm):
		metrics = {'mse': 0}
		for videos_start, videos_end in tqdm(islice(test_loader, config['DEBUG_NUM_BATCHES']), desc=f'Evaluating on Test Dataset'):
			# Both videos_start and videos_end have shape (B, 10, 64, 64)
			# We need to 'warmup' the model using the posterior, then predict last 10 frames with prior
			# Run the posterior scan for the first 10 frames, yielding the deter state to feed to the imagine function
			batch_size, length = videos_start.shape[:2]
			deter = jnp.zeros((batch_size, config['DETER_DIM']))
			deter, post_outs = rssm.observe(deter, videos_start)

			# Run the prior scan for the final 10 frames, and measure the difference
			_, prior_outs = rssm.imagine(deter, length)
			batch_metrics = compute_metrics(prior_outs['recon_obs'], videos_end)
			metrics['mse'] += batch_metrics['mse']

		# Create the comparison using the first 5 samples from the dataset
		# Ground truth data (20, 64, 64) on the top row, predicted data on the bottom row
		videos_start, videos_end = next(iter(test_loader))
		comparison = make_video_comparison(
			videos_start,
			videos_end,
			post_outs['recon_obs'],
			prior_outs['recon_obs'],
			config
		)

		metrics = {key: val / config['DEBUG_NUM_BATCHES'] for key, val in metrics.items()}
		return metrics, comparison

	# We can create the rng streams for the prior and post here (I think)
	# Each module (Post and Prior) maintains its own stream of different rngs
	jit_eval_model = jax.jit(nn.apply(eval_model, rssm))
	return jit_eval_model({'params': params}, rngs={'sample': rng})


def train_and_evaluate(config):
	rng = jax.random.PRNGKey(0)
	rng, params_rng, sample_rng, eval_rng = jax.random.split(rng, 4)
	train_loader, test_loader = load_moving_mnist(config)
	rssm = RSSM()
	dummy_deter, dummy_obs = (
		jnp.zeros((1, config['DETER_DIM'])),
		jnp.zeros((1, 20, 64, 64, 1))  # (B, L, H, W, C)
	)
	params = rssm.init({'params': params_rng, 'sample': sample_rng}, dummy_deter, dummy_obs, method=rssm.observe)['params']
	ts = train_state.TrainState.create(
		apply_fn=rssm.apply,
		params=params,
		tx=optax.adam(config['LR'])
	)

	for epoch in tqdm(range(config['NUM_EPOCHS']), desc='Epochs'):
		for videos_start, videos_end in tqdm(islice(train_loader, config['DEBUG_NUM_BATCHES']), desc=f'Epoch {epoch + 1} Training'):
			videos = jnp.concat([videos_start, videos_end], axis=1)  # (B, 20, 64, 64)
			rng, sample_rng = jax.random.split(rng)
			grads = train_step(rssm, ts.params, videos, sample_rng, config)
			ts = ts.apply_gradients(grads=grads)

		metrics, comparison = eval_f(rssm, ts.params, test_loader, eval_rng, config)
		print(f"Eval epoch: {epoch+1}, MSE: {metrics['mse']:.4f}")
		save_video(comparison, f'comparison_{epoch+1}.mp4')


if __name__ == '__main__':
	config = {
		'NUM_EPOCHS': 10,
		'LR': 1e-3,
		'BETA': 1,
		'BATCH_SIZE': 12,
		'DETER_DIM': 2048,
		'KL_BALANCE': 1,
		'FREE_BITS': 1,
		'DEBUG_NUM_BATCHES': 100,
		'COMPARISON_SAMPLES': 3,
	}
	train_and_evaluate(config)