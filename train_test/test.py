import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation, TransformReward
from torch.utils.tensorboard import SummaryWriter
from models.model import Model
from encoders.observation_encoders import *
from encoders.output_encoders import *


def test(config: dict):
    total_steps = config["episode_length"] * config["test_episode_amount"]
    writer = SummaryWriter()

    encoding_methods = {
        "rate": encode_observation_rate,
        "population": encode_observation_population,
        "temporal": encode_observation_temporal,
    }

    decoding_methods = {
        "method1": decode_output_method1,
        "rate": decode_output_rate,
        "temporal": decode_output_temporal,
        "population": decode_output_population,
        "wta": decode_output_wta,
        "vector": decode_output_vector,
    }

    input_sizes = {
        "rate": 2,
        "population": 3,
        "temporal": 2,
    }

    output_sizes = {
        "method1": 2,
        "rate": 3,
        "temporal": 2,
        "population": 3,
        "wta": 2,
        "vector": 2,
    }

    encode_function = encoding_methods.get(config["observation_encoding"])
    decode_function = decoding_methods.get(config["output_decoding"])
    input_size = input_sizes.get(config["observation_encoding"])
    output_size = output_sizes.get(config["output_decoding"])

    if encode_function is None or decode_function is None:
        raise ValueError("Invalid encoding or decoding method")
    if input_size is None or output_size is None:
        raise ValueError("Invalid input or output size")

    # Generate environment
    if config["render_test"]:
        env = gym.make(
            "Pendulum-v1",
            render_mode="human",
            g=config["gravity"],
            max_episode_steps=config["episode_length"],
        )
    else:
        env = gym.make(
            "Pendulum-v1",
            g=config["gravity"],
            max_episode_steps=config["episode_length"],
        )

    env = TransformObservation(
        env,
        lambda obs: encode_observation_rate(obs, config["time_steps_per_action"]),
    )

    # Set rewards
    reward_adjust = (np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)) / 2
    env = TransformReward(env, lambda r: r + reward_adjust)

    observation, info = env.reset()

    # Load model
    net = Model(input_size, output_size, config["time_steps_per_action"])
    state_dict = torch.load("model.pth")
    net.load_state_dict(state_dict)

    rewards = []
    total_reward = 0
    episode_length = 0
    episode_lengths = []

    for step in range(total_steps):
        spks, mem = net(observation, use_traces=False)
        action = decode_function(spks, net.spike_time)

        observation, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        episode_length += 1

        if term or trunc:
            observation, _ = env.reset()
            episode_lengths.append(episode_length)
            episode_length = 0

        rewards.append(reward)
        writer.add_scalar("Reward", reward, step)

    rewards = np.array(rewards)
    torch.save(net.state_dict(), "model.pth")
    writer.close()
    env.close()
