import torch
import snntorch.spikegen
import numpy as np

def encode_observation_method1(observation: np.ndarray, spike_time_steps: int) -> torch.Tensor:
    angle = np.arccos(observation[0]) / np.pi
    speed = (observation[2] + 8) / 16
    spk = snntorch.spikegen.rate(torch.Tensor([angle, speed]), spike_time_steps)
    return spk

def encode_observation_method2(observation: np.ndarray, spike_time_steps: int) -> torch.Tensor:
    # Another encoding method
    pass