import numpy as np

class EnvPlatoon:
    def __init__(self, num_vehicles, vehicles_len, T, num_timesteps, tau, h, ep_max, ep_max_nominal, ev_min,
                 ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold,
                 lambd, gamma, r, min_safe_distance, collision_penalty):
    
        assert r > min_safe_distance, "Standstill distance must be greater than minimum safe distance"
        assert ep_max > 0, "Maximum position gap must be positive"

        self.num_vehicles = num_vehicles
        self.vehicles_len = vehicles_len
        self.T = T
        self.num_timesteps = num_timesteps
        self.tau = tau
        self.h = h
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
        self.r = r
        self.min_safe_distance = min_safe_distance
        self.collision_penalty = collision_penalty

        self.state = None
        self.leader_actions = None
        self.current_timestep = 0

        self.rewards = []

    def reset(self, leader_actions):
        initial_velocity = np.random.uniform(self.ev_min, self.ev_max)
        initial_desired_distance = self.r + self.h * initial_velocity
        self.ep_min = self.min_safe_distance - initial_desired_distance
        
        self.state = np.array([
            np.random.uniform(self.ep_min, self.ep_max),
            np.random.uniform(self.ev_min, self.ev_max),
            np.random.uniform(self.acc_min, self.acc_max)
        ], dtype=np.float32)

        # ES: leader_actions = np.random.uniform(self.acc_min, self.acc_max, self.num_timesteps)
        self.leader_actions = leader_actions
        self.current_timestep = 0
        self.rewards = []

        return self.state
    

    def step(self, action):
        action = np.clip(action, self.u_min, self.u_max)
        prev_ep, prev_ev, prev_acc = self.state
        leader_action = self.leader_actions[self.current_timestep]

        # State transition
        next_ep = prev_ep + self.T * prev_ev - self.h * self.T * prev_acc
        next_ev = prev_ev - self.T * prev_acc + self.T * leader_action
        next_acc = (1 - (self.T / self.tau)) * prev_acc + (self.T / self.tau) * action

        current_velocity = prev_ev + self.T * leader_action
        desired_distance = self.r + self.h * current_velocity
        actual_distance = desired_distance - next_ep

        if actual_distance <= self.min_safe_distance:
            reward = self.collision_penalty
            done = True
        else:
            reward = self.compute_reward(action, next_ep, next_ev, next_acc, prev_acc)
            self.current_timestep += 1
            done = self.current_timestep >= self.num_timesteps    
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
            return r_abs
        else:
            return r_qua


    def get_cumulative_reward(self):
        cumulative_reward = 0
        for k, reward in enumerate(self.rewards):
            cumulative_reward += (self.gamma**k) * reward
        return cumulative_reward
    

    def discretize_state(self, state, num_bins):
        ep, ev, acc = state
        ep_bins = np.linspace(self.ep_min, self.ep_max, num_bins[0])
        ev_bins = np.linspace(self.ev_min, self.ev_max, num_bins[1])
        acc_bins = np.linspace(self.acc_min, self.acc_max, num_bins[2])

        discrete_ep = np.digitize(ep, ep_bins)
        discrete_ev = np.digitize(ev, ev_bins)
        discrete_acc = np.digitize(acc, acc_bins)

        return (discrete_ep, discrete_ev, discrete_acc)
    

    def discretize_action(self, action, num_bins):
        action_bins = np.linspace(self.u_min, self.u_max, num_bins)
        discrete_action = np.digitize(action, action_bins)

        return discrete_action
    

    def render(self, mode='human'):
        print(f"Current state: {self.state}")
        print(f"Current timestep: {self.current_timestep}")
        print(f"Current reward: {self.rewards[-1] if self.rewards else 'N/A'}")


    def close(self):
        self.state = None
        self.leader_actions = None
        self.current_timestep = 0
        self.rewards = []
        print("Environment closed")


    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
