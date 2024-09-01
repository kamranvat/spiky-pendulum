import torch
import numpy as np


def decode_output_method1(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    neur1 = spks[:, 0].flatten().sum()
    neur2 = spks[:, 1].flatten().sum()

    neur1 = neur1 / spike_time
    neur2 = neur2 / spike_time

    if neur1 > neur2:
        act = neur1 * 2
    else:
        act = -neur2 * 2

    return act.numpy()[np.newaxis]


def decode_output_rate(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    rates = spks.sum(dim=0) / spike_time
    rates = rates * 4 - 2
    return rates.numpy()


def decode_output_temporal(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    spike_times = (spks.cumsum(dim=0) == 1).float().argmax(dim=0)
    return spike_times.numpy()


def decode_output_population(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    population_activity = spks.sum(dim=0) / spike_time
    return population_activity.numpy()


def decode_output_wta(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    rates = spks.sum(dim=0) / spike_time
    winner = rates.argmax()
    return np.array([winner])


def decode_output_vector(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    spike_counts = spks.sum(dim=0)
    return spike_counts.numpy()
