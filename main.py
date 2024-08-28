from itertools import product
from train_test.train import train
from train_test.test import test
from config import config_dict
from imports import warnings


import gc
gc.enable()
# gc.set_debug(gc.DEBUG_LEAK)

import tracemalloc
tracemalloc.start(10)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias for `np.bool_`.",
)  # Ignore deprecation warning (gym version causes it, but nothing we can do about it)


def generate_config_combinations(observation_encodings, output_decodings, reward, learning_rates, config_dict):
    config_combinations = []
    for obs_enc, out_dec, rew_sh, lr in product(observation_encodings, output_decodings, reward, learning_rates):
        new_config = config_dict.copy()
        new_config["observation_encoding"] = obs_enc
        new_config["output_decoding"] = out_dec
        new_config["reward_shape"] = rew_sh
        new_config["learning_rate"] = lr
        config_combinations.append(new_config)
    return config_combinations


def main():
    # Generate all possible combinations of observation encodings and output decodings
    # Remove from the list if you want to exclude a certain combination
    # TODO: create compatibility table

    # Options:
    # observation_encodings = ["rate", "population", "temporal"]
    # output_decodings = ["method1", "rate", "temporal", "population", "wta", "vector"]
    # reward_shapings = ["bin", "shift", "norm_gym", "norm_one"]

    observation_encodings = ["rate"]
    output_decodings = ["rate"]
    reward_shapings = ["shift"] #, "norm_gym", "norm_one"]
    learning_rates = [0.1, 0.01] #, 0.001]

    config_combinations = generate_config_combinations(
        observation_encodings, output_decodings, reward_shapings, learning_rates, config_dict
    )


    snapshots = []
    for config in config_combinations:
        print(
            "Training combination:",
            config["observation_encoding"],
            "+",
            config["output_decoding"],
            "+",
            config["reward_shape"],
        )
        train(config)
        gc.collect()
        snapshots.append(tracemalloc.take_snapshot())
        print(
            "Testing combination:",
            config["observation_encoding"],
            "+",
            config["output_decoding"],
            "+",
            config["reward_shape"],
        )
        test(config)

        return snapshots


if __name__ == "__main__":
    snapshots = []
    snapshots.append(tracemalloc.take_snapshot())
    
    snapshots.append(main())
    snapshots.append(tracemalloc.take_snapshot())

    for idx, snap in enumerate(snapshots):
        top_stats = snap.statistics('lineno')
        print('------------------------{}------------------------'.format(idx))
        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)

        print("\n_______________________________________________________________________________________________________________\n")

        top_stats = snap.compare_to(snapshots[idx-1], 'lineno')

        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)