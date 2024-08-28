from imports import torch
from imports import snntorch
from imports import np


def encode_observation_rate(
    observation: np.ndarray, spike_time_steps: int
) -> torch.Tensor:
    angle = np.arccos(observation[0]) / np.pi
    speed = (observation[2] + 8) / 16
    spk = snntorch.spikegen.rate(torch.Tensor([angle, speed]), spike_time_steps)
    return spk


def encode_observation_population(
    observation: np.ndarray, spike_time_steps: int, population_size: int = 10
) -> torch.Tensor:
    angle = np.arccos(observation[0]) / np.pi
    speed = (observation[2] + 8) / 16

    angle_population = np.linspace(0, 1, population_size)
    speed_population = np.linspace(0, 1, population_size)

    angle_spikes = np.exp(-np.square(angle_population - angle) / (2 * 0.1**2))
    speed_spikes = np.exp(-np.square(speed_population - speed) / (2 * 0.1**2))

    angle_spikes = snntorch.spikegen.rate(torch.Tensor(angle_spikes), spike_time_steps)
    speed_spikes = snntorch.spikegen.rate(torch.Tensor(speed_spikes), spike_time_steps)

    spk = torch.cat((angle_spikes, speed_spikes), dim=0)
    return spk


def encode_observation_temporal(
    observation: np.ndarray, spike_time_steps: int
) -> torch.Tensor:
    angle = np.arccos(observation[0]) / np.pi
    speed = (observation[2] + 8) / 16
    angle_spike_time = int((1 - angle) * spike_time_steps)
    speed_spike_time = int((1 - speed) * spike_time_steps)

    spk = torch.zeros((2, spike_time_steps))
    spk[0, angle_spike_time] = 1
    spk[1, speed_spike_time] = 1
    return spk
