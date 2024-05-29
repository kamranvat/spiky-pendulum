import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformObservation

# TODOs:

# try to convert the observation space to spikes
#   observation space is ndarray of shape (3,) representing the x/y of the free end, and its angular v
#   Observation space:
#   Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)

# try to convert from spikes to action space
# Action space is ndarray with shape (1,) repre
#   Box(-2.0, 2.0, (1,), float32)


def observation_to_freq(observation, min_obs, max_obs, min_freq=1, max_freq=100):
    # don't think this function is needed actually
    observation = (observation - min_obs) / (max_obs - min_obs)
    freq = min_freq + (max_freq - min_freq) * observation


def normalize_observation(observation, min_obs, max_obs):
    """return normalized observation between 0 and 1 (for freq encoding)"""
    return (observation - min_obs) / (max_obs - min_obs)


def normalize_observation_absolute(observation, min_obs, max_obs):
    """
    for each value in the observation, return two values, one for the positive and one for the negative value
    (i.e. for the observation [0.1, -0.2, 0.3], return [0.1, 0, 0, 0.2, 0.3, 0]).
    Also normalize the values between 0 and 1.
    """
    observation = normalize_observation(observation, min_obs, max_obs)
    abs_observation = []
    for value in observation:
        if value >= 0:
            abs_observation.extend([abs(value), 0])
        else:
            abs_observation.extend([0, abs(value)])

    return abs_observation


def generate_spike(odds):
    # TODO: check if the rate is good - maybe reduce odds in here
    """generate a spike with the given odds (0-1)"""
    return np.random.rand() < odds


env = gym.make(
    "Pendulum-v1", render_mode="human", g=1
)  # default is g = 10, earth gravity is 9.8
min_obs = env.observation_space.low
max_obs = env.observation_space.high
env = TransformObservation(
    env, lambda obs: normalize_observation(obs, min_obs, max_obs)
)
normalize_observation_absolute([0.1, -0.2, 0.3], min_obs, max_obs)

observation, info = env.reset()
# try stuff out - three input neurons:
for _ in range(1000):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    # generate spikes for the input neurons
    spikes = [generate_spike(neuron) for neuron in observation]
    print(observation[2])

    if terminated or truncated:
        observation, info = env.reset()

env.close()
