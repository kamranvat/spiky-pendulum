from itertools import product
from train_test.train import train
from train_test.test import test
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias for `np.bool_`.",
)  # Ignore deprecation warning (gym version causes it, but nothing we can do about it)


def generate_config_combinations(observation_encodings, output_decodings, config_dict):
    config_combinations = []
    for obs_enc, out_dec in product(observation_encodings, output_decodings):
        new_config = config_dict.copy()
        new_config["observation_encoding"] = obs_enc
        new_config["output_decoding"] = out_dec
        config_combinations.append(new_config)
    return config_combinations


def main():
    config = {
        "time_steps_per_action": 50,
        "gravity": 0,
        "episode_length": 500,
        "train_episode_amount": 2,
        "test_episode_amount": 1,
        "render_train": False,
        "render_test": False,
        "print_act_obs": False,
        "print_before": False,
        "observation_encoding": " ",  
        "output_decoding": " ",  
    }

    observation_encodings = ["rate", "population", "temporal"]
    output_decodings = ["method1", "rate", "temporal", "population", "wta", "vector"]
    config_combinations = generate_config_combinations(
        observation_encodings, output_decodings, config
    )

    for config in config_combinations:
        print("Training combination:", config["observation_encoding"], "+", config["output_decoding"])
        train(config)
        print("Testing combination:", config["observation_encoding"], "+", config["output_decoding"])
        test(config)


if __name__ == "__main__":
    main()
