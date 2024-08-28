from models.model import Model
from imports import TransformObservation, TransformReward, SummaryWriter
from imports import torch
from imports import os, np, gym, types
from config import (
    encoding_methods,
    decoding_methods,
    reward_shaping,
    input_sizes,
    output_sizes,
    tb_train_interval,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # Suppress TensorFlow warnings (we don't use TensorFlow)
)


def train(config: dict):
    total_steps = config["episode_length"] * config["train_episode_amount"]
    writer = SummaryWriter()

    encode_function = encoding_methods.get(config["observation_encoding"])
    decode_function = decoding_methods.get(config["output_decoding"])
    reward_function = reward_shaping.get(config["reward_shape"])
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

    if isinstance(reward_function, types.FunctionType):
        env = TransformReward(env, lambda r: reward_function(r))
    else:
        env = reward_function(env)

    observation, info = env.reset()

    net = Model(input_size, output_size, config["time_steps_per_action"])
    net.set_optim(lr = config["learning_rate"])

    rewards = []
    ep_steps = 0

    for step in range(total_steps):
        spks, mem = net(observation, use_traces=True)
        action = decode_function(spks, net.spike_time)
        observation, reward, term, trunc, _ = env.step(action)
        net.optim.step(reward)
        ep_steps += 1

        if term or trunc:
            observation, _ = env.reset()
            writer.add_scalar("Episode Length", ep_steps, step)
            ep_steps = 0
            rewards = []

        rewards.append(reward)
        writer.add_scalar("Actions", action, step)

        # Track reward every tb_train_interval steps
        if step % tb_train_interval == 0:
            avg_reward = np.mean(rewards)
            median_reward = np.median(rewards)
            writer.add_scalar("Median Reward", median_reward, step)
            writer.add_scalar("Average Reward", avg_reward, step)
            rewards = []

    torch.save(net.state_dict(), "model.pth")
    writer.close()
    env.close()
