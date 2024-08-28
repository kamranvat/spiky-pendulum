from imports import np

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

def reward_shift(r):
    return r + MAX_REWARD / 2

def norm_reward_oneone(r):
    half = MAX_REWARD / 2
    return (r + half) / half