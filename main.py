from env_platoon import EnvPlatoon

num_vehicles = 2
T = 0.1
num_timesteps = 100
tau = 0.1
h = 1
ep_min = 0.2
ep_max = 2
ep_max_nominal = 15
ev_min = 0.1
ev_max = 1.5
ev_max_nominal = 10
acc_min = -2.6
acc_max = 2.6
u_min = -2.6
u_max = 2.6
a = 0.1
b = 0.1
c = 0.2
reward_threshold = -0.4483
lambd = 1
gamma = 0.99

env = EnvPlatoon(num_vehicles, T, num_timesteps, tau, h, ep_min, ep_max, ep_max_nominal, ev_min,
                 ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold,
                 lambd, gamma)
