# from imports import numpy as np

# class RSTDP:
#     def __init__(self, dt, lr,
#                  a_plus, a_minus,
#                  A_plus, A_minus,
#                  tau_a, tau_e):
#         '''
#         Define the alforithm for R-STDP

#         Args:
#             dt      -- simulation time step
#             lr      -- learnign rate for weight updates
#             a_plus  -- scaling factor for the presynaptic activity trace update
#             a_minus -- scaling factor for the postsynaptic activity trace update
#             A_plus  -- scaling factor for the positive STDP update of the eligiblity traces
#             A_minus -- scaling factor for the negative STDP update of the eligiblity traces
#             tau_a   -- the timescale of activity traces
#             tau_e   -- the timescale of eligility traces
#         '''
#         self.dt = dt
#         self.lr = lr

#         self.a_plus  = a_plus
#         self.a_minus = a_minus

#         self.A_plus  = A_plus
#         self.A_minus = A_minus

#         self.tau_a = tau_a
#         self.tau_e = tau_e

#     def update_pre_a_trace(self, trace, firing):
#         '''
#         Update a prestsynaptic activity traces.

#         Args:
#             trace  -- a vector of presynaptic activity traces
#             firing -- presynaptic firing activity
#         '''
#         d_trace = -(self.dt / self.tau_a) * trace + self.a_plus * firing
#         trace  += d_trace
#         return trace

#     def update_post_a_trace(self, trace, firing):
#         '''
#         Update a postsynaptic activity traces.

#         Args:
#             trace  -- postsynaptic activity trace
#             firing -- postsynaptic firing activity
#         '''
#         d_trace = -(self.dt / self.tau_a) * trace + self.a_minus * firing
#         trace  += d_trace
#         return trace

#     def update_elig_trace(self,
#                        pre_a_trace, pre_firing,
#                        post_a_trace, post_firing,
#                        e_trace):
#         '''
#         Update incoming and outcoming connections of a target fired neuron.

#         Args:
#             pre_a_trace  -- activity traces of presynaptic neurons
#             pre_firing   -- firing pattern of presynaptic neurons at the current time step
#             post_a_trace -- activity traces of postynaptic neurons
#             post_firing  -- firing pattern of postsynaptic neurons at the current time step
#             e_trace      -- vector with eligibility traces for all connections
#         '''
#         # positive STDP update: pre before post
#         de_pos = np.outer(post_firing, pre_a_trace) * self.A_plus
#         # negative STDP update: post before pre
#         de_neg = np.outer(post_a_trace, pre_firing) * self.A_minus
#         # the resulting STDP update
#         stdp_update = de_pos - de_neg

#         # change eligibility traces
#         e_trace += -(self.dt / self.tau_e) * e_trace + stdp_update
#         return e_trace

#     def update_weights(self, weights, reward, e_trace):
#         weights += e_trace * reward * self.lr
#         weights = np.clip(weights, 0, 20)
#         return weights

import torch
import warnings

class RSTDP(torch.optim.Optimizer):
    def __init__(self, params, time_steps : int, lr: float = 1e-3, **kwargs):
        '''
        Define the R-STDP optimizer.

        Args:
            params  -- model parameters to optimize
            time_steps -- total simulation time of the network, must be set
            lr      -- learning rate for weight updates
            dt      -- simulation time step
            a_plus  -- scaling factor for the presynaptic activity trace update
            a_minus -- scaling factor for the postsynaptic activity trace update
            A_plus  -- scaling factor for the positive STDP update of the eligibility traces
            A_minus -- scaling factor for the negative STDP update of the eligibility traces
            tau_a   -- the timescale of activity traces
            tau_e   -- the timescale of eligibility traces
        '''
        defaults = dict(
            lr = lr,
            dt = kwargs.get('dt', 5),
            a_plus = kwargs.get('a_plus', 0.008),
            a_minus = kwargs.get('a_minus', 0.0096),
            A_plus = kwargs.get('A_plus', 3.),
            A_minus = kwargs.get('A_minus', 0.),
        )

        tau_a = kwargs.get('tau_a', 20)
        tau_e = kwargs.get('tau_e', 40)

        defaults['pref_a'] = defaults['dt'] / tau_a
        defaults['pref_e'] = defaults['dt'] / tau_e

        # TODO: implement checks
        # e.g. lr > 0, dt > 0, a_plus etc >0 
        # self.arg = defaults
        self.cur_step = 1
        self.time_steps = time_steps

        super(RSTDP, self).__init__(params, defaults)


        pre_trace = []
        post_trace = []
        e_trace = []

        # param_groups is of structure: list[dict[list]]
        # the outermost list contains the groups
        # the dict contains the 'params' key, where the bias and the weights are 
        # the inner list contains the weights and the biases, as parameter tensors

        for group in self.param_groups:
            group_pre = []
            group_post = []
            group_e = []
            for param in group['params']:
                # try:
                #     group['params'][1]
                #     warnings.warn('When using this optimiser your Model shouldn\'t contain biases', RuntimeWarning)
                #     # TODO: warning call needs to be adjusted
                # except:
                #     pass

                 # we're only interested in the weights, the biases are not being trained
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


    # def update_pre_a_trace(self, trace: torch.Tensor, firing: torch.Tensor, group: dict) -> torch.Tensor:
    #     trace += -group['pref_a'] * trace + group['a_plus'] * firing
    #     return trace
    # # self.pre_trace[idx] = self.update_pre_a_trace(self.pre_trace[idx], pre_firing)

    # def update_post_a_trace(self, trace: torch.Tensor, firing: torch.Tensor, group: dict) -> torch.Tensor:
    #     trace += -group['pref_a'] * trace + group['a_minus'] * firing
    #     return trace
    # # self.post_trace[idx] = self.update_post_a_trace(self.post_trace[idx], post_firing)


    def update_e_trace(self, pre_firing: list, post_firing: list) -> None:
        '''
        TODO: Docstring   
        needs to be called in the forward loop of the model
        '''

        if self.cur_step > self.time_steps:
            raise RuntimeError()
        
        if type(pre_firing[0]) != list:
            raise ValueError(
                f'Expected pre_firing to be of the shape [param_groups, layers] \n\r \
                Often this can be resolved by passing update_e_trace([pre_firings],...) to the function.')
        
        if type(post_firing[0]) != list:
            raise ValueError(
                f'Expected post_firing to be of the shape [param_groups, layers] \n\r \
                Often this can be resolved by passing update_e_trace(...,[post_firing]) to the function.')

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
                
                # # the resulting STDP update
                stdp_update = de_pos - de_neg

                # change eligibility traces
                e_trace_temp = - group['pref_e'] * self.e_trace[idx][i][:,:,self.cur_step - 1] + stdp_update

                # catch very small values and set them to 0
                # e_trace_temp[torch.logical_and(e_trace_temp <= 1e-20, e_trace_temp >= -1e-20)] = 0
                self.e_trace[idx][i][:,:,self.cur_step] = e_trace_temp # .clamp(-1e30, 1e30) 
                # for some godforsaken reason the eligibility trace explodes here, maybe ask farbod if it should be between 0 and 1

        
        self.cur_step += 1

        


    def step(self, reward: torch.Tensor, closure = None) -> None:
        
        '''
        Applies the RSTDP learning rule based on the rewards

        Args:
            reward  -- reward for the (current) time step
            firings -- dict of structure {pre0: value, post0: value, pre1: value, post1: value ...}
            traces  -- dict of structure {pre0: value, post0: value, pre1: value, post1: value ...}
        '''
        loss = None
        self.cur_step = 1

        if type(reward) not in [torch.Tensor]:
            try:
                reward = torch.tensor(reward)
            except:
                raise RuntimeError(f'Couldn\'t convert reward to tensor. Got {type(reward)}')

        for idx, group in enumerate(self.param_groups):
            for i, p in enumerate(group['params']):

                e_trace = self.e_trace[idx][i][:,:,:self.time_steps - 1] * reward * group['lr']
                # adjusting the weights
                p.data += e_trace.sum(dim = 2)

                # very crude regularisation
                p.data = torch.clamp(p.data, -20, 20)
                if (p.data.isnan().any() == True).item():
                    raise RuntimeError(f'{p.data=} turned to nan')
                
                # if i == 2: breakpoint()

                

        return loss
