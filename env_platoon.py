import numpy as np

class EnvPlatoon:
    def __init__(self, num_vehicles, vehicles_length, num_timesteps, T, h, tau, ep_max, ep_max_nominal,
                 ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold, lambd,
                 env_gamma, r, leader_min_speed, leader_max_speed, pattern_name, min_safe_distance, collision_penalty):

        assert r > min_safe_distance, "Standstill distance must be greater than minimum safe distance"
        assert ep_max > 0, "Maximum position gap must be positive"

        # Platoon params
        self.num_vehicles = num_vehicles
        self.vehicles_length = vehicles_length
        self.num_timesteps = num_timesteps
        self.T = T
        self.h = h
        self.tau = tau

        # State limits
        self.ep_max = ep_max
        self.ep_max_nominal = ep_max_nominal
        self.ev_max = ev_max
        self.ev_max_nominal = ev_max_nominal
        self.acc_min, self.acc_max = acc_min, acc_max
        self.u_min, self.u_max = u_min, u_max

        # Reward params
        self.a = a
        self.b = b
        self.c = c
        self.reward_threshold = reward_threshold
        self.lambd = lambd
        self.env_gamma = env_gamma

        # Safety params
        self.r = r
        self.leader_min_speed = leader_min_speed / 3.6  # Converting to m/s
        self.leader_max_speed = leader_max_speed / 3.6
        self.min_safe_distance = min_safe_distance
        self.collision_penalty = collision_penalty

        self.state = None
        self.leader_actions = None
        self.current_timestep = 0
        self.rewards = []
        self.collision_step = None
        
        self.leader_velocity = None
        self.agent_velocity = None
        self.desired_distance = None
        self.actual_distance = None

    def reset(self, leader_actions, pattern_name=None):
        # Leader velocity initialization
        leader_initial_velocity = np.random.uniform(self.leader_min_speed, self.leader_max_speed)
        
        # Initial velocity error (ev) generation
        initial_ev = np.random.uniform(-self.ev_max, self.ev_max)
        agent_initial_velocity = leader_initial_velocity - initial_ev
        
        # Initial distances computation
        initial_desired_distance = self.r + self.h * agent_initial_velocity + self.vehicles_length
        initial_ep = np.random.uniform(-self.ep_max, self.ep_max)
        initial_actual_distance = initial_desired_distance + initial_ep
        
        # Random initial agent acceleration
        initial_acc = np.random.uniform(self.acc_min, self.acc_max)

        # Initial state
        self.state = np.array([initial_ep, initial_ev, initial_acc], dtype=np.float32)
        self.leader_velocity = leader_initial_velocity
        self.agent_velocity = agent_initial_velocity
        self.desired_distance = initial_desired_distance
        self.actual_distance = initial_actual_distance
        self.collision_step = None

        # Leader actions pattern setup
        self.leader_actions = leader_actions
        self.pattern_name = pattern_name
        self.current_timestep = 0
        self.leader_acc = leader_actions[self.current_timestep]
        self.rewards = []

        return self.state

    def step(self, action):
        # Updating leader actual acceleration
        self.leader_acc = (1 - (self.T / self.tau)) * self.leader_acc + (self.T / self.tau) * self.leader_actions[self.current_timestep]

        # State transition
        prev_ep, prev_ev, prev_acc = self.state
        next_ep = prev_ep + self.T * prev_ev - self.h * self.T * prev_acc
        next_ev = prev_ev - self.T * prev_acc + self.T * self.leader_acc
        next_acc = (1 - (self.T / self.tau)) * prev_acc + (self.T / self.tau) * action

        # Updating velocities and distances
        self.leader_velocity += self.T * self.leader_acc
        self.agent_velocity = self.leader_velocity - next_ev
        self.desired_distance = self.r + self.h * self.agent_velocity + self.vehicles_length
        self.actual_distance = self.desired_distance + next_ep

        # Reward computation
        if self.actual_distance < self.min_safe_distance + self.vehicles_length:
            collision_penalty = self.collision_penalty * (self.actual_distance - self.min_safe_distance - self.vehicles_length)
            reward = self.compute_reward(action, next_ep, next_ev, prev_acc) - collision_penalty
            if self.collision_step is None:    
                self.collision_step = self.current_timestep
        else:
            reward = self.compute_reward(action, next_ep, next_ev, prev_acc)
        
        self.current_timestep += 1
        done = self.current_timestep >= self.num_timesteps
        
        self.rewards.append(reward)
        self.state = np.array([next_ep, next_ev, next_acc], dtype=np.float32)

        return self.state, reward, done, {}

    def compute_reward(self, action, next_ep, next_ev, prev_acc):
        # Normalization of errors and action
        ep_norm = next_ep / self.ep_max_nominal
        ev_norm = next_ev / self.ev_max_nominal
        u_norm = action / self.u_max
        
        # Jerk computation and normalization
        jerk = (action - prev_acc) / self.tau
        jerk_norm = jerk / (2 * self.acc_max / self.T)

        # Absolute Reward
        r_abs = -(np.abs(ep_norm) + self.a * np.abs(ev_norm) + 
                 self.b * np.abs(u_norm) + self.c * np.abs(jerk_norm))
        
        # Quadratic Reward
        r_qua = -self.lambd * ((next_ep)**2 + self.a * ((next_ev)**2) + 
                              self.b * ((action)**2) + self.c * ((jerk * self.T)**2))

        # Huber-like reward
        return r_qua if r_abs >= self.reward_threshold else r_abs

    def get_cumulative_reward(self):
        """Computes Discounted Cumulative Reward"""
        return sum((self.env_gamma**k) * r for k, r in enumerate(self.rewards)) / len(self.rewards)

    def discretize_state(self, state, num_bins):
        """Discretizes state for Tabular Q-Learning"""
        ep, ev, acc = state
        ep_bins = np.linspace(-self.ep_max, self.ep_max, num_bins[0])
        ev_bins = np.linspace(-self.ev_max, self.ev_max, num_bins[1])
        acc_bins = np.linspace(self.acc_min, self.acc_max, num_bins[2])

        return (np.digitize(ep, ep_bins), 
                np.digitize(ev, ev_bins), 
                np.digitize(acc, acc_bins))

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