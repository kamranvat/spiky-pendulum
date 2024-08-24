from itertools import product
from train_test.train import train
from train_test.test import test
import warnings
from config import config_dict

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
    # Generate all possible combinations of observation encodings and output decodings
    # Remove from the list if you want to exclude a certain combination

    observation_encodings = ["rate", "population", "temporal"]
    output_decodings = ["method1", "rate", "temporal", "population", "wta", "vector"]
    config_combinations = generate_config_combinations(
        observation_encodings, output_decodings, config_dict
    )

    for config in config_combinations:
        print(
            "Training combination:",
            config["observation_encoding"],
            "+",
            config["output_decoding"],
        )
        train(config)
        print(
            "Testing combination:",
            config["observation_encoding"],
            "+",
            config["output_decoding"],
        )
        test(config)


if __name__ == "__main__":
    main()
