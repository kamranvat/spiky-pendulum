# Spiky Pendulum
_Project for the course "Modelling Synaptic Plasticity", Summer Semester of 2024, University of Osnabrück_

## Goals
To build a Gymnasium-Based RL experiment that learns to solve the Pendulum-V1 task, and to compare performance of different encoding/decoding schemes for the network.

## Installation

The installation has been tested in Ubuntu. No guarantees for a working solution on Windows/Mac systems.
To set up the conda environment, follow these steps:

1. **Clone the repository**:
    Navigate to a directory where you want to clone the project into, then use
    ```sh
    git clone https://github.com/kamranvat/synaptic_plasticity_project.git
    cd synaptic_plasticity_project
    ```

2. **Create the conda environment**:
    ```sh
    conda env create -f environment.yml
    ```

3. **Activate the conda environment**:
    ```sh
    conda activate snn
    ```

## Running the Project

To run the main script, use the following command:

```sh
python main.py
```

## Tracking the Progress or view Results

To launch tensorboard, navigate to the `synaptic_plasticity_project` folder, activate conda environment and enter following command:
```sh
tensorboard --logdir=runs
```
To view tensorboard you can open the url provided by tensorboard in the console.
The project will run and test each combination specified in main.py, which means that the runs displayed in tensorboard alternate between training performance and test performance.

## Explanation of Files

- config.py: Contains configuration settings and lookups for the project, including en- and decoding methods, reward-shaping methods, input and output sizes, and training/testing parameters.
- environment.yml: Specifies the dependencies and environment settings required to run the project. Used to create a conda environment.
- main.py: The main script to run the project. It generates training configurations, that are then passed to the training and testing processes.
- rstdp.py: Implements the Reward-modulated Spike-Timing-Dependent Plasticity (RSTDP) optimizer used in the model.
- visualisation.py: Contains functions to visualize actions and rewards from CSV files. 

- data/: Directory containing data files, like csvs.
- encoders/: Directory containing python files defining encoding and decoding functions as well as reward-shaping functions.
- models/: Directory containing model definitions and related code.
- obsolete/: Directory for old files.
- train_test/: Directory containing the train.py and test.py files. They define the respective function, including loading models, initialising gym environements and logging with tensorboard.


## Known issues:

- Leaking Memory
- 