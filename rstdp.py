from imports import numpy as np

class RSTDP:
    def __init__(self, dt, lr,
                 a_plus, a_minus,
                 A_plus, A_minus,
                 tau_a, tau_e):
        '''
        Define the alforithm for R-STDP

        Args:
            dt      -- simulation time step
            lr      -- learnign rate for weight updates
            a_plus  -- scaling factor for the presynaptic activity trace update
            a_minus -- scaling factor for the postsynaptic activity trace update
            A_plus  -- scaling factor for the positive STDP update of the eligiblity traces
            A_minus -- scaling factor for the negative STDP update of the eligiblity traces
            tau_a   -- the timescale of activity traces
            tau_e   -- the timescale of eligility traces
        '''
        self.dt = dt
        self.lr = lr

        self.a_plus  = a_plus
        self.a_minus = a_minus

        self.A_plus  = A_plus
        self.A_minus = A_minus

        self.tau_a = tau_a
        self.tau_e = tau_e

    def update_pre_a_trace(self, trace, firing):
        '''
        Update a prestsynaptic activity traces.

        Args:
            trace  -- a vector of presynaptic activity traces
            firing -- presynaptic firing activity
        '''
        d_trace = -(self.dt / self.tau_a) * trace + self.a_plus * firing
        trace  += d_trace
        return trace

    def update_post_a_trace(self, trace, firing):
        '''
        Update a postsynaptic activity traces.

        Args:
            trace  -- postsynaptic activity trace
            firing -- postsynaptic firing activity
        '''
        d_trace = -(self.dt / self.tau_a) * trace + self.a_minus * firing
        trace  += d_trace
        return trace

    def update_e_trace(self,
                       pre_a_trace, pre_firing,
                       post_a_trace, post_firing,
                       e_trace):
        '''
        Update incoming and outcoming connections of a target fired neuron.

        Args:
            pre_a_trace  -- activity traces of presynaptic neurons
            pre_firing   -- firing pattern of presynaptic neurons at the current time step
            post_a_trace -- activity traces of postynaptic neurons
            post_firing  -- firing pattern of postsynaptic neurons at the current time step
            e_trace      -- vector with eligibility traces for all connections
        '''
        # positive STDP update: pre before post
        de_pos = np.outer(post_firing, pre_a_trace) * self.A_plus
        # negative STDP update: post before pre
        de_neg = np.outer(post_a_trace, pre_firing) * self.A_minus
        # the resulting STDP update
        stdp_update = de_pos - de_neg

        # change eligibility traces
        e_trace += -(self.dt / self.tau_e) * e_trace + stdp_update
        return e_trace

    def update_weights(self, weights, reward, e_trace):
        weights += e_trace * reward * self.lr
        weights = np.clip(weights, 0, 20)
        return weights