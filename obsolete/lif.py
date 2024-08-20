from imports import numpy as np

class LIF:
    def __init__(self, dt=1.0, V_thr=-55, V_noise=0.0, t_refr=3):
        '''
        Simulate the activity of a LIF neuron.

        Args:
            V_thr     -- threshold for neuron firing
            V_noise   -- random noise added to the signal
            t_refr    -- refractory period
            dt        -- time step for the simulation
        Constant params:
            g_L     -- leak of the conductance (inverse membrane resistance)
            tau_m   -- membrane time constant: how quickly membrane voltage returns to the resting state
            E_L     -- resting potential (equal to voltage reset after spike)
        '''
        self.dt    = dt
        self.E_L   = -65
        self.tau_m = 10.0
        self.g_L   = 10.0

        self.V_thr   = V_thr
        self.V_noise = V_noise
        self.t_refr  = t_refr

    def time_step(self, V, I, refr):

        # Voltage update
        noise = np.random.rand() * self.V_noise

        # insert your formula for the membrane potential (voltage) update
        dV = noise + (-(V - self.E_L) + I / self.g_L) * self.dt / self.tau_m
        # integrate the above update
        V += dV

        # refractory
        V[refr > 0] = self.E_L  # set V to resting potential if recently after spike
        refr[refr > 0] -= 1     # decrease the refractory period counter over time

        fired = V > self.V_thr
        # what happens to the neurons whose membrane potential exceeded a threshold?
        V[fired] = self.E_L
        refr[fired] = self.t_refr / self.dt  # initialize a refractory period counter for the neurons fired
        return V, fired, refr