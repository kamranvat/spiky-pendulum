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
    # don't think this function is needed actually
    observation = (observation - min_obs) / (max_obs - min_obs)
    freq = min_freq + (max_freq - min_freq) * observation

def normalize_obs(observation, min_obs, max_obs):
    """return normalized observation between 0 and 1 (for freq encoding)"""
    return (observation - min_obs) / (max_obs - min_obs)

# if we use this with frequency encoding, we will need to use 3 input neurons firing at 50% to represent 0 (0% for -1, 100% for 1), 
#   instead of 6 input neurons which actually go from 0% for 0 to 100% for 1. 
def normalize_obs_space(observation, min_obs, max_obs):
    """normalize each value of the observation space with its respective high and low
    
    Args:
        observation (np.ndarray): observation space

    Returns:
        list: normalized observation space
    """
    return [normalize_obs(observation[i], min_obs[i], max_obs[i]) for i in range(len(observation))]

def generate_spike(odds):
    """generate a spike with the given odds (0-1)"""
    return np.random.rand() < odds


env = gym.make('Pendulum-v1', render_mode="human", g=1) # default is g = 10, earth gravity is 9.8
observation, info = env.reset()
min_obs = env.observation_space.low
max_obs = env.observation_space.high


# try stuff out - three input neurons:
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    # represent the neurons as colors in the pygame window, light up when a spike is generated

    # normalize the action space
    observation = normalize_obs_space(observation, min_obs, max_obs)

    # generate spikes for the input neurons
    spikes = [generate_spike(neuron) for neuron in observation]
    print(spikes)



    if terminated or truncated:
        observation, info = env.reset()

env.close()
