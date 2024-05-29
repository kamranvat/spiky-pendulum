import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformObservation, RescaleAction, NormalizeObservation

# TODOs:

# try to convert the observation space to spikes
#   observation space is ndarray of shape (3,) representing the x/y of the free end, and its angular v
#   Observation space:
#   Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)

# try to convert from spikes to action space
# Action space is ndarray with shape (1,) repre
#   Box(-2.0, 2.0, (1,), float32)


def normalize_observation(observation, min_obs, max_obs):
    """return normalized observation between 0 and 1 (for freq encoding)"""
    return (observation - min_obs) / (max_obs - min_obs)


def normalize_observation_signed(observation, min_obs, max_obs):
    """return normalized observation between -1 and 1 (for freq encoding)"""
    return 2 * (observation - min_obs) / (max_obs - min_obs) - 1


def normalize_observation_absolute(observation, min_obs, max_obs):
    """
    for each value in the observation, return two values, one for the positive and one for the negative value
    (i.e. for the observation [0.1, -0.2, 0.3], return [0.1, 0, 0, 0.2, 0.3, 0]).
    Also normalize the values between 0 and 1.
    """
    observation = normalize_observation_signed(observation, min_obs, max_obs)
    abs_observation = []
    for value in observation:
        if value >= 0:
            abs_observation.extend([abs(value), 0])
        else:
            abs_observation.extend([0, abs(value)])

    return abs_observation


def generate_spike(odds: float) -> bool:
    # TODO: check if the rate is good - maybe reduce odds in here
    # didn't we do that with a poisson process instead?
    """generate a spike with the given odds (0-1)"""
    return np.random.rand() < odds


def pretty_print_spikes(spikes):
    """print the spikes in a human readable way"""
    print("".join([" X " if spike else "   " for spike in spikes]))


# Generate environment:
env = gym.make("Pendulum-v1", render_mode="human", g=1)  # default: g=10
min_obs = env.observation_space.low
max_obs = env.observation_space.high

# that might be helpful
# env = RescaleAction(env, 0, 1);

env = TransformObservation(
    env, lambda obs: normalize_observation_absolute(obs, min_obs, max_obs)
)

observation, info = env.reset()
# try stuff out - three input neurons:
for _ in range(1000):
    action = env.action_space.sample()  # random policy
    observation, reward, terminated, truncated, info = env.step(action)

    # generate spikes for the input neurons
    spikes = [generate_spike(neuron) for neuron in observation]
    pretty_print_spikes(spikes)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
