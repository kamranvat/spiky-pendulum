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


def obs_transf(box, min_fire: int = 100, max_fire: int = 100) -> np.array:
    # since the box is given as 
    #   | obs           | Min | Max
    # 0 | x = cos(theta)| -1  |  1
    # 1 | y = sin(theta)| -1  |  1
    # 2 | angl. veloc.  | -8  |  8
    # we can also represent theta as arccos(x) # returns radians, rad2deg 
    # and if we use the angle (log scaled) to generate spike rates, that might be helpful
    # 180deg should be low firing rate, 0 deg should be max. 
    # we could try clipping it at 180deg and use the second column with -arccos(x) for angles >

    if min_fire < 0 or max_fire < 0:
        raise ValueError(f'min_fire and max_fire should be a positive integer.\nGot {min_fire=} and {max_fire=}')
    
    angle = np.arccos(box[0])
    
    if angle <= np.pi:
        angle = (np.pi - angle) / np.pi
    else:
        angle = (angle - np.pi) / np.pi

    angle = np.square(angle)

    # angle = np.exp(angle) * max_fire

    ##  or we just square it - idea is to get a function that looks like an exponential function, but in range (0,1)


    # still have to think about to what to do with the angular velocity
    
    return np.array([angle,box[2]])
    


def pretty_print_spikes(input_spikes, output_spikes):
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
    min_obs = env.observation_space.low
    max_obs = env.observation_space.high

    # that might be helpful

    env = TransformObservation(
        env, 
        lambda obs: obs_transf(obs)
    )

    observation, info = env.reset()
    max_angl = []

    # try stuff out - for six input neurons:
    for _ in range(1000):
        action = env.action_space.sample()  # random policy
        # generate random spikes of two output neurons
        # output_spikes = [generate_spike(0.5), generate_spike(0.5)]
        # action = spikes_to_action(output_spikes, env.action_space.low, env.action_space.high)
        observation, reward, terminated, truncated, info = env.step(action)
        max_angl.append(observation[0])

        # generate spikes for the input neurons (for current timestep)
        # input_spikes = [generate_spike(neuron) for neuron in observation]
        # pretty_print_spikes(input_spikes, output_spikes)

        if terminated or truncated:
            observation, info = env.reset()

    print(max(max_angl))
    env.close()
