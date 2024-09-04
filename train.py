import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from tqdm import tqdm
from nets import RSSM
from dataset import load_moving_mnist


sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)

@jax.vmap
def mse(preds, targets):
	return optax.squared_error(preds, targets)

def train_step(rssm, params, videos, z_rng, config):
	def loss_fn(params):
		# prepare initial state and use model to predict output
		batch_size = videos.shape[0]
		deter = jnp.zeros((batch_size, config["DETER_DIM"]))
		outs = rssm.apply({"params": params}, deter, videos, method=rssm.observe)
		
		# compute losses
		mse_loss = mse(outs["recon_obs"], videos).mean()
		return mse_loss
		
		# TODO: incorporate distribution losses and implement STE
		# dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))
		# rep = self._dist(post).kl_divergence(self._dist(sg(prior)))
		# dyn = jnp.maximum(dyn, config["FREE_BITS"])
		# rep = jnp.maximum(rep, config["FREE_BITS"])
		# kld_loss = dyn + config["KL_BALANCE"] * rep
		# loss = mse_loss + config["BETA"] * kld_loss
		# return loss

	grads = jax.grad(loss_fn)(params)
	return grads

def train_and_evaluate(config):
	rng = jax.random.PRNGKey(0)
	rng, init_rng, eval_rng = jax.random.split(rng, 3)
	train_loader, test_loader = load_moving_mnist(config)
	rssm = RSSM()
	dummy_deter, dummy_obs = (
		jnp.zeros((1, config["DETER_DIM"])),
		jnp.zeros((1, 20, 64, 64, 1))  # (B, L, H, W, C)
	)
	params = rssm.init(init_rng, dummy_deter, dummy_obs, method=rssm.observe)["params"]
	ts = train_state.TrainState.create(
		apply_fn=rssm.apply,
		params=params,
		tx=optax.adam(config["LR"])
	)

	for epoch in tqdm(range(config["NUM_EPOCHS"]), desc="Epochs"):
		for videos_start, videos_end in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
			videos = jnp.concat([videos_start, videos_end], axis=1)  # (B, 20, 64, 64)
			rng, z_rng = jax.random.split(rng)
			grads = train_step(rssm, ts.params, videos, z_rng, config)
			ts = ts.apply_gradients(grads=grads)

		# TODO: save videos and print metrics after each epoch

if __name__ == "__main__":
	config = {
		"NUM_EPOCHS": 10,
		"LR": 1e-3,
		"BETA": 0.001,
		"BATCH_SIZE": 1,
		"DETER_DIM": 2048,
	}
	train_and_evaluate(config)