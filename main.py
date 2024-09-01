from itertools import product
from train_test.train import train
from train_test.test import test
import warnings
from config import config_dict
import argparse

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias for `np.bool_`.",
)  # Ignore deprecation warning (gym version causes it, but nothing we can do about it)


def generate_config_combinations(
    observation_encodings, output_decodings, reward, learning_rates, config_dict
):
    config_combinations = []
    for obs_enc, out_dec, rew_sh, lr in product(
        observation_encodings, output_decodings, reward, learning_rates
    ):
        new_config = config_dict.copy()
        new_config["observation_encoding"] = obs_enc
        new_config["output_decoding"] = out_dec
        new_config["reward_shape"] = rew_sh
        new_config["learning_rate"] = lr
        config_combinations.append(new_config)
    return config_combinations


def main(train_only=False, test_only=False):
    # Generate all possible combinations of observation encodings and output decodings
    # Remove from the list if you want to exclude a certain combination
    # TODO: create compatibility table

    # Options:
    # observation_encodings = ["rate", "population", "temporal"]
    # output_decodings = ["method1", "rate", "temporal", "population", "wta", "vector"]
    # reward_shapings = ["bin", "shift", "norm_gym", "norm_one"]

    observation_encodings = ["rate"]
    output_decodings = ["rate"]
    reward_shapings = ["norm_one"]
    learning_rates = [0.0001]

    config_combinations = generate_config_combinations(
        observation_encodings,
        output_decodings,
        reward_shapings,
        learning_rates,
        config_dict,
    )

    for config in config_combinations:
        if not test_only:
            print(
                "Training combination:",
                config["observation_encoding"],
                "+",
                config["output_decoding"],
                "+",
                config["reward_shape"],
            )
            train(config)

        if not train_only:
            print(
                "Testing combination:",
                config["observation_encoding"],
                "+",
                config["output_decoding"],
                "+",
                config["reward_shape"],
            )
            test(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and/or testing.")
    parser.add_argument("-train", action="store_true", help="Run training only")
    parser.add_argument("-test", action="store_true", help="Run testing only")
    args = parser.parse_args()

    main(train_only=args.train, test_only=args.test)
