import snntorch.spikegen
from imports import gymnasium as gym
from imports import numpy as np

from gymnasium.wrappers import TransformObservation, TransformReward, NormalizeReward

import torch
import snntorch
from rstdp import RSTDP


# TODO:

# try to convert the observation space to spikes
#   observation space is ndarray of shape (3,) representing the x/y of the free end, and its angular v
#   Observation space:
#   Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)

# try to convert from spikes to action space
# Action space is ndarray with shape (1,) repre
#   Box(-2.0, 2.0, (1,), float32)

class Model(torch.nn.Module):
    def __init__(self, time_steps_per_action: int = 50):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        self.spike_time = time_steps_per_action

        self.con1 = torch.nn.Linear(2, 100, bias = False) # bias = False is recommended? for RSTDP optimiser
        self.lif1 = snntorch.Leaky(0.9, 0.5, surrogate_disable = True)
        self.con2 = torch.nn.Linear(100, 2, bias = False)
        self.lif2 = snntorch.Leaky(0.9, 0.5, surrogate_disable = True)
        # self.con3 = torch.nn.Linear(50, 2, bias = False)
        # self.lif3 = snntorch.Leaky(0.9, 0.5, surrogate_disable = True)
        

    def forward(self, x: torch.Tensor, use_traces: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        if use_traces and not self.optim:
            raise RuntimeError(f'You have to set the optimiser first to return traces.')
        
        mem_rec = []
        spk_rec = []
        for step in range(self.spike_time):
            cur = self.con1(x[step])
            spk1, mem = self.lif1(cur)
            cur2 = self.con2(spk1)
            spk2, mem2 = self.lif2(cur2)
            # cur3 = self.con3(spk2)
            # spk3, mem3 = self.lif3(cur3)

            mem_rec.append(mem2)
            spk_rec.append(spk2)

            if use_traces:
                self.optim.update_e_trace(
                    pre_firing = [[x[step],spk1]],
                    post_firing = [[spk1, spk2]]
                    )

        return torch.stack(spk_rec), torch.stack(mem_rec)
    
    def make_action(self, spks: torch.Tensor):
        '''
        Takes in 2 dimensional Tensor of spikes, creates action
        Args:
            spks -- Tensor, should be of shape (time_steps, num_neurons_out)
        
        '''

        neur1 = spks[:,0].flatten().sum()
        neur2 = spks[:,1].flatten().sum()

        neur1 = neur1 / self.spike_time
        neur2 = neur2 / self.spike_time

        if neur1 > neur2:
            act = neur1 * 2
            print('neuron 1 has been chosen')
        else:
            act = -neur2 * 2
        ###################
        # is it even useful to have two output neurons? - since the rstdp algorithm doesn't distinguish between which neuron did what
        # just if it's any good
        ##################

        # neuron 1 is the positive action neuron, neuron 2 the negative.
        return act.numpy()[np.newaxis]
    
    def set_optim(self, lr:float = 0.01, **kwargs):
        self.optim = RSTDP(self.parameters(), time_steps = self.spike_time, lr = lr, **kwargs)


def make_spikes(box: np.ndarray, spike_time_steps: int = 50) -> torch.Tensor:
    # should return float tensor
    angle = np.arccos(box[0]) / np.pi
    speed = (box[2] + 8) / 16
    spk = snntorch.spikegen.rate(torch.Tensor([angle,speed]), spike_time_steps)
    return spk


if __name__ == "__main__":

    # Generate environment:
    env = gym.make("Pendulum-v1", render_mode = 'human', g=1)  # default: g=10
    time_steps_per_action = 300

    env = TransformObservation(
        env, 
        lambda obs: make_spikes(obs, time_steps_per_action)
    )

    # maybe normalizing the reward around 0 is more useful
    reward_adjust = (np.square(np.pi) + 0.1 * np.square(8) + 0.001 * np.square(2)) / 2
    env = TransformReward(
        env,
        lambda r: r + reward_adjust
    )
    # env = NormalizeReward(env) # max reward was -0.01, even though thingy was upriht
    # reward should be centered around 0, maybe max 1, min -1, but most importantly for a totally upright position max

    observation, info = env.reset()

    net = Model(time_steps_per_action)
    net.set_optim() # there are a lot of kwargs here, though I set defaults


    before = net.state_dict()
    print(before)

    rewardl = []


    for i in range(500):
        # action = env.action_space.sample()  # random policy

        spks, mem = net(observation, use_traces = True)
        action = net.make_action(spks)

        # printing stuff
        if action != 0:
            print(f'act{action}')
        else: print(f'obs{observation.max().item()}') # see if there are any spikes in the observation

        observation, reward, term, trunc, _ = env.step(action)
        net.optim.step(reward)
        
        if term or trunc:
            observation, _ = env.reset()
        
        rewardl.append(reward)


    rewardl = np.array(rewardl)

    breakpoint()
    env.close()