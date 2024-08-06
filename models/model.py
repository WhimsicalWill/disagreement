# Use FLAX to create a network
# The learned world model is going to take in an observation, embed it, and predict the embedding of the next observation
# The environment is a simple 1-step env and transitions as follows:
    # 0 digit -> 0 digit
    # 1 digit -> any digit
class MLP(nn.Module):                    # create a Flax Module dataclass
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)       # shape inference
    return x

model = MLP(out_dims=10)                 # instantiate the MLP model

x = jnp.empty((4, 28, 28, 1))            # generate random data
variables = model.init(random.key(42), x)# initialize the weights
y = model.apply(variables, x)  