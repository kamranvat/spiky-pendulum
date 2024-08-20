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


def prob_to_poisson_input(probabilities: np.ndarray, spike_time_steps: int = 1) -> np.ndarray:
    '''
    ### Generate input in the form of Poisson spike process based on given probabilities.
    probabilities: 1D ndarray, with values between 0 and 1
    spike_time_steps: int >= 0
    returns: ndarray of int8 and max value of 1.
    '''
    rng = np.random.default_rng()

    poisson_trains = rng.poisson(probabilities[:,np.newaxis], (probabilities.shape[0], spike_time_steps))
    poisson_trains = poisson_trains.clip(max = 1)

    return poisson_trains.astype(np.int8)


def obs_transf(box: np.ndarray, spike_time_steps: int = 1) -> np.ndarray:
    '''
    Since the box is given as:
    |      Obs      | Min | Max |
    |      ---      | --- | --- |
    | x = cos(theta)| -1  |  1  |
    | y = sin(theta)| -1  |  1  |
    | angl. veloc.  | -8  |  8  |
    
    From that we can calculate theta with arccos(x) and use this to generate spike rates. 
    180 Degrees (straight down) should be lowest firing rate, 0 Degrees (straight up) should be max.
    To have a stronger bias towards upright, the probability to spike gets squared.

    The velocity gets normalized.
    '''

    if spike_time_steps < 0:
        raise ValueError(f'spike_time_steps must be a positive integer.\nGot {spike_time_steps=}')

    # is this smart??
    angle = np.arccos(box[0])
    
    if angle <= np.pi:
        angle = (np.pi - angle) / np.pi
    else:
        angle = (angle - np.pi) / np.pi
    
    # to encode a strong bias to upright position make the lower values smaller by squaring.
    angle = np.power(angle, 2)

    # normalize 
    velocity = (box[2] + 8) / 16  # box[2] -- 8 / 8 -- 8

    # we assume for 100 timesteps to take place in each action that can be taken - so that we can have an array with up to 100 spikes
    probs = np.array([angle,velocity])
    return prob_to_poisson_input(probs, spike_time_steps)
    



def pretty_print_spikes(input_spikes, output_spikes) -> None:
    """print the spikes in a human readable way"""
    print(
        ("  |     ")
        + "".join([" X " if spike else "   " for spike in input_spikes])
        + ("     |     ")
        + ("").join([" <- " if output_spikes[0] else "    ", " -> " if output_spikes[1] else "    "])
        + ("     |  ")
    )


if __name__ == "__main__":

    # Generate environment:
    env = gym.make("Pendulum-v1", render_mode="human", g=1)  # default: g=10
    time_steps_per_action = 1

    # that might be helpful
    env = TransformObservation(
        env, 
        lambda obs: obs_transf(obs, time_steps_per_action)
    )

    observation, info = env.reset()

    # try stuff out - for six input neurons:
    for _ in range(1000):
        action = env.action_space.sample()  # random policy
        # generate random spikes of two output neurons
        # output_spikes = [generate_spike(0.5), generate_spike(0.5)]
        # action = spikes_to_action(output_spikes, env.action_space.low, env.action_space.high)
        observation, reward, term, trunc, _ = env.step(action)
        print(f'Step: {observation}')

        # generate spikes for the input neurons (for current timestep)
        # input_spikes = [generate_spike(neuron) for neuron in observation]
        # pretty_print_spikes(input_spikes, output_spikes)

        if term or trunc:
            observation, _ = env.reset()

    env.close()
