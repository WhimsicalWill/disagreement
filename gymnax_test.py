import jax
import gymnax
from mnist_env import MNISTOneStep


jax.config.update("jax_disable_jit", True)

N_EPISODES = 1

def test_environment(env):
    env = MNISTOneStep()
    env_params = env.default_params
    rng = jax.random.PRNGKey(42)

    for _ in range(N_EPISODES):
        rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

        # Reset the environment
        obs, state = env.reset(key_reset, env_params)
        print("Initial State Label:", state.label)

        # Sample a random action
        action = env.action_space(env_params).sample(key_act)

        # Perform the step transition
        n_obs, n_state, reward, done, info = env.step(key_step, state, action, env_params)
        print(f"Next State: {n_state.label}, Reward: {reward}, Done: {done}")
        print("-" * 50)

if __name__ == "__main__":
    test_environment(0)
