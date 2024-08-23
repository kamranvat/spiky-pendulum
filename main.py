from train_test.train import train
from train_test.test import test

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias for `np.bool_`.",
)  # Ignore deprecation warning (gym version causes it, but nothing we can do about it)

if __name__ == "__main__":
    config = {
        "time_steps_per_action": 50,
        "gravity": 0,
        "episode_length": 500,
        "train_episode_amount": 2,
        "test_episode_amount": 1,
        "render_train": False,
        "render_test": True,
        "print_act_obs": False,
        "print_before": False,
        "observation_encoding": "rate",  # Choose encoding method for observation
        "output_encoding": "method1",  # Choose encoding method for output
    }

    print("Training...")
    train(config)
    print("Testing...")
    test(config)
