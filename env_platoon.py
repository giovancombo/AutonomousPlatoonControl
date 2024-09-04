import numpy as np

class EnvPlatoon:
    def __init__(self, num_vehicles, T, num_timesteps, tau, h, ep_min, ep_max, ep_max_nominal, ev_min,
                 ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold,
                 lambd, gamma):

        self.num_vehicles = num_vehicles
        self.T = T
        self.num_timesteps = num_timesteps
        self.tau = tau
        self.h = h
        self.ep_min = ep_min
        self.ep_max = ep_max
        self.ep_max_nominal = ep_max_nominal
        self.ev_min = ev_min
        self.ev_max = ev_max
        self.ev_max_nominal = ev_max_nominal
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.u_min = u_min
        self.u_max = u_max
        self.a = a
        self.b = b
        self.c = c
        self.reward_threshold = reward_threshold
        self.lambd = lambd
        self.gamma = gamma

        self.state = None
        self.leader_actions = None
        self.current_timestep = 0

        self.rewards = []

    def reset(self, leader_actions):
        self.state = np.array([
            np.random.uniform(self.ep_min, self.ep_max),
            np.random.uniform(self.ev_min, self.ev_max),
            np.random.uniform(self.acc_min, self.acc_max)
        ], dtype=np.float32)

        # ES: leader_actions = np.random.uniform(self.acc_min, self.acc_max, self.num_timesteps)
        self.leader_actions = leader_actions
        self.current_timestep = 0

        return self.state
    

    def step(self, action):        
        action = np.clip(action, self.u_min, self.u_max)
        prev_ep, prev_ev, prev_acc = self.state
        leader_action = self.leader_actions[self.current_timestep]

        # State transition
        next_ep = prev_ep + self.T * prev_ev - self.h * self.T * prev_acc
        next_ev = prev_ev - self.T * prev_acc + self.T * leader_action
        next_acc = (1 - (self.T / self.tau)) * prev_acc + (self.T / self.tau) * action

        reward = self.compute_reward(action, next_ep, next_ev, next_acc, prev_acc)
        self.rewards.append(reward)

        self.current_timestep += 1
        done = self.current_timestep >= self.num_timesteps

        self.state = np.array([next_ep, next_ev, next_acc], dtype=np.float32)

        return self.state, reward, done, {}
    

    def compute_reward(self, action, next_ep, next_ev, next_acc, prev_acc):
        # Compute jerk, absolute reward and quadratic reward
        jerk = (next_acc - prev_acc) / self.T
        r_abs = -(np.abs(next_ep/self.ep_max_nominal) + self.a * np.abs(next_ev/self.ev_max_nominal) + self.b * np.abs(action/self.u_max) + self.c * np.abs(jerk/self.acc_max))
        r_qua = -self.lambd * ((next_ep)**2 + self.a * (next_ev)**2 + self.b * (action)**2 + self.c * (jerk * self.T)**2)

        # Huber Loss
        if r_abs < self.reward_threshold:
            reward = r_abs
        else:
            reward = r_qua

        return reward
    

    def get_cumulative_reward(self):
        cumulative_reward = 0
        for k, reward in enumerate(self.rewards):
            cumulative_reward += (self.gamma**k) * reward
            
        return cumulative_reward