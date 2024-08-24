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

config_dict = {
        "time_steps_per_action": 50,
        "gravity": 0,
        "episode_length": 500,
        "train_episode_amount": 2,
        "test_episode_amount": 1,
        "render_train": False,
        "render_test": True,
        "print_act_obs": False,
        "print_before": False,
        "observation_encoding": " ",  
        "output_decoding": " ",  
    }