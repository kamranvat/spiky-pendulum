import gymnasium as gym
import numpy as np

# TODOs:

# try to convert the observation space to spikes
#   observation space is ndarray of shape (3,) representing the x/y of the free end, and its angular v
#   Observation space:
#   Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)

# try to convert from spikes to action space
# Action space is ndarray with shape (1,) repre
#   Box(-2.0, 2.0, (1,), float32)

def observation_to_freq(observation, min_obs, max_obs, min_freq=1, max_freq=100):
    # normalize the observation
    observation = (observation - min_obs) / (max_obs - min_obs)
    # convert to  (frequency)
    freq = min_freq + (max_freq - min_freq) * observation

def normalize_obs(observation, min_obs, max_obs):
    """return normalized observation between 0 and 1 (for freq encoding)"""
    return (observation - min_obs) / (max_obs - min_obs)

def normalize_obs_space(observation):
    """normalize each value of the observation space with its respective high and low"""
    return [normalize_obs(observation[i], observation.low[i], observation.high[i]) for i in range(len(observation))]

def generate_spike(odds):
    """generate a spike with the given odds (0-1)"""
    return np.random.rand() < odds


env = gym.make('Pendulum-v1', render_mode="human", g=1) # default is g = 10
observation, info = env.reset()

# print the observation space, and for each value in it print the min and max
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)

# idea for how to convert to spikes:



for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
