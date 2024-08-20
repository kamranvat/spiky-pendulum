import torch
import numpy as np

def encode_output_method1(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    neur1 = spks[:, 0].flatten().sum()
    neur2 = spks[:, 1].flatten().sum()

    neur1 = neur1 / spike_time
    neur2 = neur2 / spike_time

    if neur1 > neur2:
        act = neur1 * 2
    else:
        act = -neur2 * 2

    return act.numpy()[np.newaxis]

def encode_output_method2(spks: torch.Tensor, spike_time: int) -> np.ndarray:
    # Another encoding method
    pass