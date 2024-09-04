import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax


treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)


class OneHotDist(distrax.OneHotCategorical):

    def __init__(self, logits=None, probs=None):
        super().__init__(logits, probs)

    def sample(self, sample_shape=(), seed=None):
        sample = super().sample(sample_shape=sample_shape, seed=seed)
        # Straight-through estimator for sampling
        sample = sg(sample) + (self.probs - sg(self.probs))
        return sample


class Encoder(nn.Module):
	"""VAE Encoder."""

	depth: int = 64
	mults: tuple = (1, 3, 2)
	kernel: int = 3
	stride: int = 2

	@nn.compact
	def __call__(self, x):
		depths = [self.depth * m for m in self.mults]
		for d in depths:
			x = nn.Conv(
				features=d,
				kernel_size=(self.kernel, self.kernel), 
				strides=(self.stride, self.stride)
			)(x)
			x = nn.gelu(x)
		# x has shape (B, 4, 4, 128)
		x = x.reshape((x.shape[0], -1))  # (B, 4*4*128)
		x = nn.Dense(128, name='fc1')(x)  # (B, 128)
		x = nn.relu(x)
		return x


class Decoder(nn.Module):
	"""VAE Decoder."""

	@nn.compact
	def __call__(self, deter, stoch):
		stoch = stoch.reshape(stoch.shape[0], -1)  # flatten to (B, S * C)
		x = jnp.concat([deter, stoch], axis=-1)  # (B, D + S * C)
		x = nn.Dense(features=16*16*64)(x)  # (B, 16*16*64)
		x = nn.relu(x)
		x = x.reshape((-1, 16, 16, 64))  # (B, 16, 16, 64)
		x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # (B, 32, 32, 32)
		x = nn.relu(x)
		x = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # (B, 64, 64, 16)
		x = nn.relu(x)
		x = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')(x)  # (B, 64, 64, 1)
		x = nn.sigmoid(x)  # Ensure the output is in the interval [0, 1]
		return x


class RSSM(nn.Module):
	"""Full RSSM model."""

	stoch: int = 32
	classes: int = 32
	gru_hidden_dim: int = 64

	def setup(self):
		# define the networks
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.gru = nn.GRUCell(self.gru_hidden_dim)
		self.prior = Prior(self.stoch, self.classes)
		self.posterior = Posterior(self.stoch, self.classes)

		# define the prior scan on top of the PriorCell module
		self.prior_scan = nn.scan(
            PriorCell,
			variable_broadcast='params',
			in_axes=1,
			out_axes=1,
			split_rngs={'params': False}
		)(
			self.decoder,
			self.gru,
			self.prior
		)

		# define the posterior scan on top of the PostCell module
		self.post_scan = nn.scan(
            PostCell,
			variable_broadcast='params',
			in_axes=1,
			out_axes=1,
			split_rngs={'params': False}
		)(
			self.encoder,
			self.decoder,
			self.gru,
			self.prior,
			self.posterior
		)

	def observe(self, deter, obs):
		_, outs = self.post_scan(deter, obs)
		return outs

	def imagine(self, deter):
		_, outs = self.prior_scan(deter)
		return outs


class Prior(nn.Module):

	stoch: int
	classes: int

	@nn.compact
	def __call__(self, deter):
		batch_size = deter.shape[0]
		logits = nn.Dense(self.stoch * self.classes)(deter)  # (B, S * C)
		logits = logits.reshape((batch_size, self.stoch, self.classes))  # (B, S, C)
		return logits


class PriorCell(nn.Module):

	decoder: nn.Module
	gru: nn.Module
	prior: nn.Module

	def __call__(self, deter):
		# predict the prior so we can compute the intrinsic reward
		prior_logits = self.prior(deter)
		# sample from the posterior to get stoch state
		# TODO: gracefully supply different rng for each iteration of the scan
		stoch = OneHotDist(prior_logits).sample(seed=jax.random.PRNGKey(0))
		stoch = stoch.reshape((stoch.shape[0], -1))
		# reconstruct the obs using compact state
		recon_obs = self.decoder(deter, stoch)
		# apply the gru to produce the next deter state
		_, deter = self.gru(deter, stoch)
		# return a tuple of carry, out
		return (
			deter, 
			dict(
				prior_logits=prior_logits,
				recon_obs=recon_obs,
			)
		)
	

class Posterior(nn.Module):

	stoch: int
	classes: int

	@nn.compact
	def __call__(self, deter, embed):
		batch_size = deter.shape[0]
		x = jnp.concat([deter, embed], axis=-1)  # (B, D + E)
		logits = nn.Dense(self.stoch * self.classes)(x)  # (B, S * C)
		logits = logits.reshape((batch_size, self.stoch, self.classes))  # (B, S, C)
		return logits


class PostCell(nn.Module):

	encoder: nn.Module
	decoder: nn.Module
	gru: nn.Module
	prior: nn.Module
	posterior: nn.Module

	def __call__(self, deter, obs):
		'''
		This is the function that takes in a previous deter state 
		and the next observation and produces the next deter state.

		Args:
			deter: The deterministic features associated with the current state
			obs: The image observation of the next timestep
		Returns:
			The compact state of the next step in a dictionary
		'''
		# predict the prior so we can train it via the loss function
		prior_logits = self.prior(deter)
		# get the lower dimensional encoding of the image
		embed = self.encoder(obs)
		# use both to predict the logits using the posterior
		post_logits = self.posterior(deter, embed)
		# sample from the posterior to get stoch state
		# TODO: gracefully supply different rng for each iteration of the scan
		stoch = OneHotDist(post_logits).sample(seed=jax.random.PRNGKey(0))
		stoch = stoch.reshape((stoch.shape[0], -1))
		# reconstruct the obs using compact state
		recon_obs = self.decoder(deter, stoch)
		# apply the gru to produce the next deter state
		_, deter = self.gru(deter, stoch)
		# return a tuple of carry, out
		# print(f"{deter.shape=}")
		# print(f"{obs.shape=}")
		# print(f"{prior_logits.shape=}")
		# print(f"{embed.shape=}")
		# print(f"{post_logits.shape=}")
		# print(f"{stoch.shape=}")
		# print(f"{recon_obs.shape=}")
		# print(f"{deter.shape=}")
		return (
			deter, 
			dict(
				post_logits=post_logits,
				prior_logits=prior_logits,
				recon_obs=recon_obs,
			)
		)