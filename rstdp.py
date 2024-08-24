import torch
import warnings
import numpy as np

class RSTDP(torch.optim.Optimizer):
    def __init__(self, params, time_steps : int, lr: float = 1e-3, **kwargs):
        '''
Define the R-STDP optimizer.
Arguments:
|    Name   | Type  | Description                                                           | Defaults |
|-----------|:-----:|-----------------------------------------------------------------------|:--------:|
|params     | None  | model parameters to optimize                                          |     -    |
|time_steps | int   | total simulation time of the network, must be set                     |     -    |
|lr         | float | learning rate for weight updates,                                     | 1e-3     |
|dt         | int   | simulation time step,                                                 | 5        |
|a_plus     | float | scaling factor for the presynaptic activity trace update,             | 8e-3     |
|a_minus    | float | scaling factor for the postsynaptic activity trace update,            | 9.6e-4   |
|A_plus     | float | scaling factor for the positive STDP update of the eligibility traces | 3        |
|A_minus    | float | scaling factor for the negative STDP update of the eligibility traces | 0        |
|tau_a      | int   | the timescale of activity traces,                                     | 20       |
|tau_e      | int   | the timescale of eligibility traces,                                  | 40       |
        '''
        defaults = dict(
            lr = lr,
            a_plus = kwargs.get('a_plus', 0.008),
            a_minus = kwargs.get('a_minus', 0.0096),
            A_plus = kwargs.get('A_plus', 3.),
            A_minus = kwargs.get('A_minus', 1.),
        )

        dt = kwargs.get('dt', 5)
        tau_a = kwargs.get('tau_a', 20)
        tau_e = kwargs.get('tau_e', 40)

        defaults['pref_a'] = dt / tau_a
        defaults['pref_e'] = dt / tau_e

        self.time_steps = time_steps
        self.cur_step = 1

        # checking if user inputs are correct.
        if time_steps <= 0 or not isinstance(time_steps, (int, np.int_, torch.int)):
            raise ValueError(f'Time steps should be a postive integer value. Got {time_steps=}.')
        
        if defaults['lr'] <= 0:
            raise ValueError(f'Learning rate should be a positive float value. Got {defaults["lr"]=}.')
        
        if dt <= 0 or not isinstance(dt, (int, np.int_, torch.int)):
            raise ValueError(f'dt should be a positive integer value. Got {dt=}.')
        
        if defaults['a_plus'] <= 0:
            raise ValueError(f'a_plus should be a positive value. Got {defaults["a_plus"]=}.') 
        
        if defaults['a_minus'] <= 0:
            raise ValueError(f'a_minus should be a positive value. Got {defaults["a_minus"]=}.') 
        
        if defaults['A_plus'] <= 0:
            raise ValueError(f'A_plus should be a positive value. Got {defaults["A_plus"]=}.')
        
        if defaults['A_minus'] <= 0:
            raise ValueError(f'A_minus should be a positive value. Got {defaults["A_minus"]=}.')
        
        if tau_a <= 0 or not isinstance(tau_a, (int, np.int_, torch.int)):
            raise ValueError(f'Tau A should be a positive integer value. Got {tau_a=}.')
        
        if tau_e <= 0 or not isinstance(tau_e, (int, np.int_, torch.int)):
            raise ValueError(f'Tau E should be a positive integer value. Got {tau_e=}.')

        super(RSTDP, self).__init__(params, defaults)

        # param_groups is of structure: list[dict[list]]
        # the outermost list contains the groups
        # the dict contains the 'params' key, where the bias and the weights are 
        # the inner list contains the weights and the biases, as parameter tensors

        # creating traces
        pre_trace = []
        post_trace = []
        e_trace = []

        for group in self.param_groups:
            group_pre = []
            group_post = []
            group_e = []

            for param in group['params']:
                group_pre.append(torch.zeros((param.shape[1],), dtype = torch.float32))
                group_post.append(torch.zeros((param.shape[0],), dtype = torch.float32))
                group_e.append(torch.zeros((param.shape[0], param.shape[1], self.time_steps + 1)))

            pre_trace.append(group_pre)
            post_trace.append(group_post)
            e_trace.append(group_e)

        # save traces
        self.pre_trace = pre_trace
        self.post_trace = post_trace
        self.e_trace = e_trace
        
        return None


    def update_e_trace(self, pre_firing: list, post_firing: list) -> None:
        '''
        Update of the eligibility traces. Has to be updated every timestep of the Network.
    
        Arguments:    
            pre_firing:  list containing a list of all the spikes that happend at pre-synaptic
            post_firing: list containing a list of all the spikes that happened at post-synaptic
        '''

        if self.cur_step > self.time_steps:
            raise RuntimeError(
                'Time steps got messed up, this shouldn\'t ever happen. \
                Check if you called update_e_trace() without calling step() afterwards'
                )
        
        if type(pre_firing[0]) != list:
            raise ValueError(
                f'Expected pre_firing to be of the shape [param_groups, layers] \n\r \
                Often this can be resolved by passing update_e_trace([pre_firings],...) to the function.'
                )
        
        if type(post_firing[0]) != list:
            raise ValueError(
                f'Expected post_firing to be of the shape [param_groups, layers] \n\r \
                Often this can be resolved by passing update_e_trace(...,[post_firing]) to the function.'
                )

        for idx, group in enumerate(self.param_groups):
            for i in range(len(group['params'])):

                # positive STDP update: pre before post
                dtrace = - group['pref_a'] * self.pre_trace[idx][i] + group['a_plus'] * pre_firing[idx][i]
                self.pre_trace[idx][i] = self.pre_trace[idx][i] + dtrace

                de_pos = torch.outer(post_firing[idx][i], self.pre_trace[idx][i]) * group['A_plus']
                

                # negative STDP update: post before pre
                dtrace = - group['pref_a'] * self.post_trace[idx][i] + group['a_minus'] * post_firing[idx][i]
                self.post_trace[idx][i] = self.post_trace[idx][i] + dtrace

                de_neg = torch.outer(self.post_trace[idx][i], pre_firing[idx][i]) * group['A_minus']
                
                
                # the resulting STDP update
                stdp_update = de_pos - de_neg

                # change eligibility traces
                self.e_trace[idx][i][:,:,self.cur_step] = - group['pref_e'] * self.e_trace[idx][i][:,:,self.cur_step - 1] + stdp_update

        self.cur_step += 1
        return None

        


    def step(self, reward: torch.Tensor, closure = None) -> None:
        
        '''
        Applies the RSTDP learning rule based on the rewards.

        Args:
            reward  -- reward for the (current) time step
        '''

        self.cur_step = 1

        # try converting reward into tensor
        if not isinstance(reward, torch.Tensor):
            try:
                reward = torch.tensor(reward)
            except:
                raise ValueError(f'Could not convert reward to tensor. Got {type(reward)}')

        if reward.isnan().any() == True:
            raise ValueError(f'Got reward that is NaN.\n\r{reward=}')

        for idx, group in enumerate(self.param_groups):
            for i, p in enumerate(group['params']):

                # calculating the adjustment of the weights
                e_trace = self.e_trace[idx][i][:,:,1:self.time_steps] * reward * group['lr']
               
                # weight update, summing the time steps together 
                p.data.add_(e_trace.sum(dim = 2))

                if p.data.isnan().any() == True:
                    raise ValueError(
                        f'Debug Error: Due to some error in the calculations (maybe dividing by 0?), \
                        the weights now contain NaNs \n\r{p.data=}'
                        )
                    