import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation, TransformReward
from torch.utils.tensorboard import SummaryWriter
from models.model import Model
from encoders.observation_encoders import *
from encoders.output_encoders import *
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # Suppress TensorFlow warnings (we don't use TensorFlow)
)


def train(config: dict):
    total_steps = config["episode_length"] * config["train_episode_amount"]
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
    if config["render_train"]:
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

    # Adjust observation space based on the encoding method
    env = TransformObservation(
        env,
        lambda obs: encode_function(obs, config["time_steps_per_action"]),
    )

    reward_adjust = (np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)) / 2
    env = TransformReward(env, lambda r: r + reward_adjust)

    observation, info = env.reset()

    net = Model(input_size, output_size, config["time_steps_per_action"])
    net.set_optim()

    rewards = []
    ep_steps = 0

    for step in range(total_steps):
        spks, mem = net(observation, use_traces=True)
        action = decode_function(spks, net.spike_time)
        ep_steps += 1

        observation, reward, term, trunc, _ = env.step(action)
        net.optim.step(reward)

        if term or trunc:
            observation, _ = env.reset()
            rewards = []

        rewards.append(reward)
        writer.add_scalar("Reward", reward, step)

    rewards = np.array(rewards)
    torch.save(net.state_dict(), "model.pth")
    writer.close()
    env.close()
