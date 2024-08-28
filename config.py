# This file contains all the configuration for the training and testing of the model

# All encoding and decoding methods are imported here
from encoders.observation_encoders import (
    encode_observation_rate,
    encode_observation_population,
    encode_observation_temporal,
)

from encoders.output_encoders import (
    decode_output_method1,
    decode_output_rate,
    decode_output_temporal,
    decode_output_population,
    decode_output_wta,
    decode_output_vector,
)

from encoders.reward_shaping import (
    bin_reward,
    reward_shift,
    norm_reward_oneone
)
from gymnasium.wrappers.normalize import NormalizeReward

# Encoding and decoding method names
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

reward_shaping = {
    "bin": bin_reward,
    "shift": reward_shift,
    "norm_gym": NormalizeReward,
    "norm_one": norm_reward_oneone
}

# Input sizes for each encoding method (number of neurons)
# TODO these are placeholder values
input_sizes = {
    "rate": 2,
    "population": 3,
    "temporal": 2,
}

# Output sizes for each decoding method (number of neurons)
# TODO these are placeholder values
output_sizes = {
    "method1": 2,
    "rate": 3,
    "temporal": 2,
    "population": 3,
    "wta": 2,
    "vector": 2,
}

# Configuration for training and testing
config_dict = {
    "time_steps_per_action": 50,
    "gravity": 0,
    "episode_length": 500,
    "train_episode_amount": 25,
    "test_episode_amount": 1,
    "render_train": False,
    "render_test": True,
    "print_act_obs": False,
    "print_before": False,
    "record_actions": False,
    "runs": 0,
    "observation_encoding": " ",    # Choose in main.py
    "output_decoding": " ",         # Choose in main.py
    "reward_shape": " ",            # Choose in main.py
    "learning_rate": 0,             # Choose in main.py
}

# Interval for logging to TensorBoard (in steps)
tb_train_interval = 250
tb_test_interval = 10