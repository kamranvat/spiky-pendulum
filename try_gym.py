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
    for each value in the observation, generate two values, one for the positive and one for the negative absolute value
    (i.e. for the observation [0.1, -0.2, 0.3], convert to [0.1, 0,   0, 0.2,   0.3, 0]).
    Then normalize the values between 0 and 1 and return the result.
    """
    observation = normalize_observation_signed(observation, min_obs, max_obs)
    abs_observation = []
    for value in observation:
        if value >= 0:
            abs_observation.extend([abs(value), 0])
        else:
            abs_observation.extend([0, abs(value)])

    return abs_observation


def spikes_to_action(spikes, min_action, max_action):
    """convert output of two spiking neurons to action space"""
    # we could add more fine-grained control with more neurons or by using membrane voltage
    if spikes[0] and not spikes[1]:
        action = min_action
    elif not spikes[0] and spikes[1]:
        action = max_action
    else:
        action = np.zeros_like(min_action)

    return action


def generate_spike(odds: float) -> bool:
    # TODO: check if this is the right way to generate spikes
    # didn't we do that with a poisson process instead? -> yes we did
    """generate a spike with the given odds (0-1)"""
    return np.random.rand() < odds


def pretty_print_spikes(input_spikes, output_spikes):
    """print the spikes in a human readable way"""
    print(
        ("  |     ")
        + "".join([" X " if spike else "   " for spike in input_spikes])
        + ("     |     ")
        + ("").join([" <- " if output_spikes[0] else "    ", " -> " if output_spikes[1] else "    "])
        + ("     |  ")
    )



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

# try stuff out - for six input neurons:
for _ in range(1000):
    action = env.action_space.sample()  # random policy
    # generate random spikes of two output neurons
    output_spikes = [generate_spike(0.5), generate_spike(0.5)]
    action = spikes_to_action(output_spikes, env.action_space.low, env.action_space.high)
    observation, reward, terminated, truncated, info = env.step(action)

    # generate spikes for the input neurons (for current timestep)
    input_spikes = [generate_spike(neuron) for neuron in observation]
    pretty_print_spikes(input_spikes, output_spikes)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
