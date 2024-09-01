import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def vis_action(filename: str = "data/csv/action.csv", index_col: str = 'Step', part: int = -1) -> None:
    df = pd.read_csv(filename, index_col = index_col)
    cols = df.columns
    w = len(cols)
    
        
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(7,5)
    for idx, col in enumerate(cols):
        ax.plot(df.index[:part], df[col][:part])
    plt.show()


def vis_reward(filename: str = "data/csv/reward.csv", index_col: str = "Step", part: int = -1, different_plots: bool = False) -> None:
    df = pd.read_csv(filename, index_col = index_col)
    cols = df.columns
    w = len(cols)

    if different_plots:
        fig, ax = plt.subplots(1, w)
        fig.set_size_inches(7*w,5)

    else: 
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(7,5)

    for idx, col in enumerate(cols):
        if different_plots:
            ax[idx].plot(df.index[:part], df[col][:part])
        else:
            ax.plot(df.index[:part], df[col][:part])
    
    plt.show()
    

if __name__ == "__main__":
    path = "data/csv/"
    files = ["promising_actions.csv", "promising_reward.csv", "sinus_reward.csv", "vastly_different.csv"]

    for file in files:
        if "action" in file:
            vis_action(path + file, part = 100)
        if "reward" in file:
            vis_reward(path + file)
        if "vastly" in file:
            vis_reward(path + file)