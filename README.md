# Spiky Pendulum
_Project for the course "Modelling Synaptic Plasticity", Summer Semester of 2024, University of Osnabrück_

## Goals
To build a Gymnasium-Based RL experiment that actually learns something, and to compare performance of different encoding/decoding schemes for the network.

## Installation

To set up the conda environment, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create the conda environment**:
    ```sh
    conda env create -f environment.yml
    ```

3. **Activate the conda environment**:
    ```sh
    conda activate <environment-name>
    ```

## Running the Project

To run the main script, use the following command:

```sh
python main.py
```
To track training progress / view test results, you can open the url provided by tensorboard in the console.
The project will run and test each combination specified in main.py, which means that the runs displayed in tensorboard alternate between training performance and test performance.
