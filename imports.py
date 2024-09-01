# python built in stuff
import types
import os
import warnings

# numpy
import numpy as np

# torch
import torch
from torch.utils.tensorboard import SummaryWriter
import snntorch
import snntorch.spikegen

# gym
import gymnasium as gym
from gymnasium.wrappers import TransformObservation, TransformReward
from gymnasium.wrappers.normalize import NormalizeReward

# # encoders
# from encoders.observation_encoders import (
#     encode_observation_rate,
#     encode_observation_population,
#     encode_observation_temporal,
# )

# # decoders
# from encoders.output_encoders import (
#     decode_output_method1,
#     decode_output_rate,
#     decode_output_temporal,
#     decode_output_population,
#     decode_output_wta,
#     decode_output_vector,
# )

# # rewards
# from encoders.reward_shaping import (
#     bin_reward,
#     reward_shift,
#     norm_reward_oneone
# )

# # training and testing
# from train_test.train import train
# from train_test.test import test

# # models
# from models.big_model import Model as BigModel
# from models.medium_model import Model as MediumModel
# from models.model import Model as Model

# # optimiser
# from rstdp import RSTDP

# # config
# from config import (
#     encoding_methods,
#     decoding_methods,
#     reward_shaping,
#     input_sizes,
#     output_sizes,
#     config_dict,
#     tb_train_interval,
#     tb_test_interval
# )