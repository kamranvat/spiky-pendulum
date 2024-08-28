import numpy as np
from scipy.stats import norm

MAX_REWARD = np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)

def bin_reward(r):
    if r < -12:
        return -0.5
    elif r < -8:
        return -0.25
    elif r < -4:
        return 0
    elif r < 0:
        return 0.25
    elif r >= 0:
        return 0.5

def norm_reward_shift(r):
    return r + MAX_REWARD / 2

def norm_reward_gauss(r):
    reward = r / -MAX_REWARD

    # sanity check:
    if reward >= 1 or reward <= 0:
        raise ValueError("Reward calculation is wrong.")

    return norm.ppf(reward)

def norm_reward_sigm(r):
    reward = (r + MAX_REWARD) / MAX_REWARD
    return 1 / (1 + np.exp(-reward))