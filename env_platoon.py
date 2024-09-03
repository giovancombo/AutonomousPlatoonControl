import numpy as np

class EnvPlatoon:
    def __init__(self, state, num_vehicles, leader_action, initial_ep, initial_ev, initial_acc,
                 timestep_length, num_timesteps, tau, h, ep_max, ev_max, a_min, a_max, u_min, u_max,
                 a, b, c, reward_threshold):
        
        self.state = None
        self.num_vehicles = num_vehicles
        self.leader_action = leader_action
        self.initial_ep = initial_ep
        self.initial_ev = initial_ev
        self.initial_acc = initial_acc
        self.timestep_length = timestep_length
        self.num_timesteps = num_timesteps
        self.tau = tau
        self.h = h
        self.ep_max = ep_max
        self.ev_max = ev_max
        self.a_min = a_min
        self.a_max = a_max
        self.u_min = u_min
        self.u_max = u_max
        self.a = a
        self.b = b
        self.c = c
        self.reward_threshold = reward_threshold

    def reset(self):
        self.state = np.array([self.initial_ep, self.initial_ev, self.initial_acc], dtype = np.float32)
        return self.state
