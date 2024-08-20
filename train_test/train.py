import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation, TransformReward
from torch.utils.tensorboard import SummaryWriter
from models.model import Model
from encoders.observation_encoders import encode_observation_method1, encode_observation_method2
from encoders.output_encoders import encode_output_method1, encode_output_method2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings (we don't use TensorFlow)

def train(config: dict):
    total_steps = config["episode_length"] * config["train_episode_amount"]
    writer = SummaryWriter()

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

    if config["observation_encoding"] == "method1":
        env = TransformObservation(
            env, lambda obs: encode_observation_method1(obs, config["time_steps_per_action"])
        )
        input_size = 2  # Adjust based on the encoding method
    elif config["observation_encoding"] == "method2":
        env = TransformObservation(
            env, lambda obs: encode_observation_method2(obs, config["time_steps_per_action"])
        )
        input_size = 3  # Adjust based on the encoding method

    reward_adjust = (np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)) / 2
    env = TransformReward(env, lambda r: r + reward_adjust)

    observation, info = env.reset()

    if config["output_encoding"] == "method1":
        output_size = 2  # Adjust based on the encoding method
    elif config["output_encoding"] == "method2":
        output_size = 3  # Adjust based on the encoding method

    net = Model(input_size, output_size, config["time_steps_per_action"])
    net.set_optim()

    rewards = []
    ep_steps = 0

    for step in range(total_steps):
        spks, mem = net(observation, use_traces=True)
        if config["output_encoding"] == "method1":
            action = encode_output_method1(spks, net.spike_time)
        elif config["output_encoding"] == "method2":
            action = encode_output_method2(spks, net.spike_time)
        ep_steps += 1

        observation, reward, term, trunc, _ = env.step(action)
        net.optim.step(reward)

        if term or trunc:
            observation, _ = env.reset()
            rewards = []

        rewards.append(reward)
        writer.add_scalar('Reward', reward, step)

    rewards = np.array(rewards)
    torch.save(net.state_dict(), "model.pth")
    writer.close()
    env.close()