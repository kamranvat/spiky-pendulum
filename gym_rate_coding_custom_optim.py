import snntorch.spikegen
from imports import gymnasium as gym
from imports import numpy as np

from gymnasium.wrappers import TransformObservation, TransformReward, NormalizeReward

import torch
import snntorch
from rstdp import RSTDP


# TODO:

# try to convert the observation space to spikes
#   observation space is ndarray of shape (3,) representing the x/y of the free end, and its angular v
#   Observation space:
#   Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)

# try to convert from spikes to action space
# Action space is ndarray with shape (1,) repre
#   Box(-2.0, 2.0, (1,), float32)

# Debug / view
PRINT_ACT_OBS = False
PRINT_BEFORE = False
VERBOSE = True


class Model(torch.nn.Module):
    def __init__(self, time_steps_per_action: int = 50):
        super().__init__()

        # Try cuda, then mps, then cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.spike_time = time_steps_per_action
        self.optim = None

        self.con1 = torch.nn.Linear(
            2, 100, bias=False
        )  # bias = False is recommended? for RSTDP optimiser
        self.lif1 = snntorch.Leaky(0.9, 0.5, surrogate_disable=True)
        self.con2 = torch.nn.Linear(100, 2, bias=False)
        self.lif2 = snntorch.Leaky(0.9, 0.5, surrogate_disable=True)
        # self.con3 = torch.nn.Linear(50, 2, bias = False)
        # self.lif3 = snntorch.Leaky(0.9, 0.5, surrogate_disable = True)

        if VERBOSE:
            print(f"Model initialized - using device: {self.device}")

    def forward(
        self, x: torch.Tensor, use_traces: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if use_traces and self.optim is None:
            raise RuntimeError(f"You have to set the optimiser first to return traces.")

        mem_rec = []
        spk_rec = []
        for step in range(self.spike_time):
            cur = self.con1(x[step])
            spk1, mem = self.lif1(cur)
            cur2 = self.con2(spk1)
            spk2, mem2 = self.lif2(cur2)
            # cur3 = self.con3(spk2)
            # spk3, mem3 = self.lif3(cur3)

            mem_rec.append(mem2)
            spk_rec.append(spk2)

            if use_traces:
                self.optim.update_e_trace(
                    pre_firing=[[x[step], spk1]], post_firing=[[spk1, spk2]]
                )

        return torch.stack(spk_rec), torch.stack(mem_rec)

    def make_action(self, spks: torch.Tensor):
        """
        Takes in 2 dimensional Tensor of spikes, creates action
        Args:
            spks -- Tensor, should be of shape (time_steps, num_neurons_out)

        """

        neur1 = spks[:, 0].flatten().sum()
        neur2 = spks[:, 1].flatten().sum()

        neur1 = neur1 / self.spike_time
        neur2 = neur2 / self.spike_time

        if neur1 > neur2:
            act = neur1 * 2
            # print("neuron 1 has been chosen")
        else:
            act = -neur2 * 2
        ###################
        # is it even useful to have two output neurons? - since the rstdp algorithm doesn't distinguish between which neuron did what
        # just if it's any good
        ##################

        # neuron 1 is the positive action neuron, neuron 2 the negative.
        return act.numpy()[np.newaxis]

    def set_optim(self, lr: float = 0.01, **kwargs):
        self.optim = RSTDP(
            self.parameters(), time_steps=self.spike_time, lr=lr, **kwargs
        )


def make_spikes(box: np.ndarray, spike_time_steps: int = 50) -> torch.Tensor:
    """generate spikes from observation space

    Args:
        box (np.ndarray): observation space
        spike_time_steps (int, optional): amount of spike trace timesteps per gym environment timestep. Defaults to 50.

    Returns:
        torch.Tensor: spike tensor
    """
    # should return float tensor
    angle = np.arccos(box[0]) / np.pi
    speed = (box[2] + 8) / 16
    spk = snntorch.spikegen.rate(torch.Tensor([angle, speed]), spike_time_steps)
    return spk


def train(config: dict):
    total_steps = config["episode_length"] * config["train_episode_amount"]

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

    if VERBOSE:
        print(
            f"Training for {config['train_episode_amount']} * {config['episode_length']} = {total_steps} steps..."
        )

    env = TransformObservation(
        env, lambda obs: make_spikes(obs, config["time_steps_per_action"])
    )

    # maybe normalizing the reward around 0 is more useful
    reward_adjust = (np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)) / 2
    env = TransformReward(env, lambda r: r + reward_adjust)

    observation, info = env.reset()

    net = Model(config["time_steps_per_action"])
    net.set_optim()  # there are a lot of kwargs here, though I set defaults

    before = net.state_dict()
    if config["print_before"]:
        print(before)

    rewards = []
    ep_steps = 0

    for _ in range(total_steps):
        spks, mem = net(observation, use_traces=True)
        action = net.make_action(spks)
        ep_steps += 1

        # printing stuff
        if config["print_act_obs"]:
            if action != 0:
                print(f"act{action}")
            else:
                print(
                    f"obs{observation.max().item()}"
                )  # see if there are any spikes in the observation

        observation, reward, term, trunc, _ = env.step(action)
        net.optim.step(reward)

        if term or trunc:
            observation, _ = env.reset()
            print(f"Episode ended ({ep_steps} steps). Mean reward: {np.mean(rewards)}")
            rewards = []

        rewards.append(reward)

    rewards = np.array(rewards)

    print(f"Training complete. Mean reward: {rewards.mean()}")
    torch.save(net.state_dict(), "model.pth")

    env.close()


def test(config: dict):

    total_steps = config["episode_length"] * config["test_episode_amount"]

    # Get model, load in state dict from training
    net = Model(config["time_steps_per_action"])
    state_dict = torch.load("model.pth")
    net.load_state_dict(state_dict)

    if VERBOSE:
        print("Loading model for testing...")

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
        env, lambda obs: make_spikes(obs, config["time_steps_per_action"])
    )

    # set rewards like in training for now
    reward_adjust = (np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)) / 2
    env = TransformReward(env, lambda r: r + reward_adjust)

    observation, info = env.reset()

    rewards = []
    total_reward = 0
    episode_length = 0
    episode_lengths = []

    for _ in range(total_steps):
        spks, mem = net(observation, use_traces=False)  # no traces needed for testing
        action = net.make_action(spks)

        # printing stuff
        if config["print_act_obs"]:
            if action != 0:
                print(f"act{action}")
            else:
                print(
                    f"obs{observation.max().item()}"
                )  # see if there are any spikes in the observation

        observation, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        episode_length += 1

        if term or trunc:
            observation, _ = env.reset()
            episode_lengths.append(episode_length)
            episode_length = 0

        rewards.append(reward)

    rewards = np.array(rewards)

    # TODO - add some print statements to see what's going on
    #     average_reward = total_reward / total_steps

    #     print(f"Total Reward: {total_reward}")
    #     print(f"Episode Lengths: {episode_lengths}")
    #     print(f"Avg. Episode Length: {np.mean(episode_lengths)}")
    #     print(f"Average Reward: {average_reward}")

    torch.save(net.state_dict(), "model.pth")

    env.close()


if __name__ == "__main__":

    config = {
        "time_steps_per_action": 50,
        "gravity": 0,  # default: g=10
        "episode_length": 500,
        "train_episode_amount": 20,
        "test_episode_amount": 1,
        "render_train": False,
        "render_test": True,
        "print_act_obs": False,
        "print_before": False,
    }

    train(config)
    test(config)
