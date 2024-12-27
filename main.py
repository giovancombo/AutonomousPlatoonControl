def set_seeds(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
import torch
import threading
import time
import wandb

from env_platoon import EnvPlatoon
from visualizer import PlatooningVisualizer
from agent import DQNAgent, TabularQAgent

TABULAR_QL = True          # True to use Tabular Q-Learning, False to use DQL
num_episodes = 15000
num_timesteps = 100         # Number of timesteps per episode

num_vehicles = 2
vehicles_length = 4         # Length of each vehicle in meters
T = 0.1                     # Duration of each timestep in seconds
tau = 0.1                   # Time constant for the vehicle dynamics
h = 1.5                     # Time to bridge the distance to the leader while proceeding at constant speed
r = 2                       # Standstill distance: minimum safety distance to the leader while stopped
min_safe_distance = 0.3     # Distance to the leader in meters that results in collision
collision_penalty = 0       # Penalty for collision

ep_max = 2                  # Maximum position gap
ep_max_nominal = 15         # Maximum position gap nominal value (for normalization)
ev_max = 1.5                # Maximum velocity gap
ev_max_nominal = 10         # Maximum valocity gap nominal value (for normalization)
acc_min = -2.6              # Minumum limit for acceleration
acc_max = 2.6               # Maximum limit for acceleration

u_min = -2.6                # Minumum limit for control input
u_max = 2.6                 # Maximum limit for control input

a = 0.1                     # Reward Weight for the velocity gap term
b = 0.1                     # Reward Weight for the control input term
c = 0.2                     # Reward Weight for the jerk term
reward_threshold = -0.4483  # Threshold for switching between absolute and quadratic reward
lambd = 5e-3                # Scale factor for quadratic reward
env_gamma = 1               # Discount factor for cumulative reward (1 = No discount)

leader_min_speed = 50       # Minimum initial leader velocity in km/h
leader_max_speed = 50       # Maximum initial leader velocity in km/h

state_size = 3
hidden_size = [256, 128]                                # Hidden layer sizes for the DQN
discrete_actions = True                                 # True for discretized actions, False for continuous actions
action_size = 1 if not discrete_actions else 10         # Size of the action space
state_bins = (10, 10, 10)                               # Size of the state space bins (only for Tabular Q-Learning)

lr = 1e-1                   # Learning rate
agent_gamma = 0.95          # Discount factor for future rewards
soft_update_tau = 0.01      # Soft Update coefficient for the Target Network
max_grad_norm = 1.0         # Gradient Clipping Norm

epsilon = 1.0               # Initial epsilon for ε-greedy strategy
eps_decay = 0.9995          # Decay rate for epsilon
min_epsilon = 0.02          # Minimum epsilon value

buffer_size = 300000        # Size of the experience replay buffer
batch_size = 512
update_freq = 80           # Frequency of the Target Q-Network update

window_size = 100           # Size of the window for calculating the average score
visualization_freq = 20000  # Frequency of visualization of episodes
log_freq = 200              # Frequency of logging states to WandB

validation_freq = 25
validation_episodes = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "num_vehicles": num_vehicles,
    "vehicles_length": vehicles_length,
    "min_safe_distance": min_safe_distance,
    "num_timesteps": num_timesteps,
    "T": T,
    "tau": tau,
    "h": h,
    "ep_max": ep_max,
    "ep_max_nominal": ep_max_nominal,
    "ev_max": ev_max,
    "ev_max_nominal": ev_max_nominal,
    "acc_min": acc_min,
    "acc_max": acc_max,
    "u_min": u_min,
    "u_max": u_max,
    "a": a,
    "b": b,
    "c": c,
    "reward_threshold": reward_threshold,
    "lambd": lambd,
    "r": r,
    "collision_penalty": collision_penalty,
    "leader_min_speed": leader_min_speed,
    "leader_max_speed": leader_max_speed,
    "hidden_size": hidden_size,
    "action_size": action_size,
    "state_bins": state_bins,
    "lr": lr,
    "agent_gamma": agent_gamma,
    "soft_update_tau": soft_update_tau,
    "max_grad_norm": max_grad_norm,
    "epsilon": epsilon,
    "eps_decay": eps_decay,
    "min_epsilon": min_epsilon,
    "buffer_size": buffer_size,
    "batch_size": batch_size,
    "update_freq": update_freq,
    "validation_freq": validation_freq,
    "validation_episodes": validation_episodes,
}

run_name = "DQN" if not TABULAR_QL else "TAB"
run_name = run_name + f"_speed{leader_max_speed}_{num_timesteps}steps_{str(time.time())[-4:]}"

class LeaderPatternGenerator:
    def __init__(self, num_timesteps, T, acc_max=2.6):        
        self.num_timesteps = num_timesteps
        self.T = T
        self.acc_max = acc_max
        self.patterns = {'uniform': self.uniform_pattern,
                        'uniform_acc': self.uniform_acc_pattern,
                        'uniform_dec': self.uniform_dec_pattern,
                        'sinusoidal': self.sinusoidal_pattern,
                        'stop_and_go': self.stop_and_go_pattern,
                        'acc_dec_sequence': self.acc_dec_sequence,
                        'random_changes': self.random_changes_pattern}
        
    def uniform_pattern(self):
        return np.zeros(self.num_timesteps)
        
    def uniform_acc_pattern(self):
        """Constant positive acceleration"""
        return np.ones(self.num_timesteps) * self.acc_max * 0.4
        
    def uniform_dec_pattern(self):
        """Constant negative acceleration"""
        return np.ones(self.num_timesteps) * -self.acc_max * 0.4
        
    def sinusoidal_pattern(self):
        t = np.linspace(0, 4*np.pi, self.num_timesteps)
        return np.sin(t) * self.acc_max * 0.3
        
    def stop_and_go_pattern(self):
        """Typical traffic pattern with stops and accelerations"""
        pattern = np.zeros(self.num_timesteps)
        segment_length = self.num_timesteps // 4
        
        pattern[:segment_length] = self.acc_max * 0.4
        pattern[2*segment_length:3*segment_length] = -self.acc_max * 0.4
        return pattern
        
    def acc_dec_sequence(self):
        pattern = np.zeros(self.num_timesteps)
        segment_length = self.num_timesteps // 5
        
        pattern[:segment_length] = self.acc_max * 0.5
        pattern[2*segment_length:3*segment_length] = -self.acc_max * 0.5
        pattern[4*segment_length:] = self.acc_max * 0.5
        return pattern
        
    def random_changes_pattern(self):
        """Smooth random changes in acceleration"""
        num_changes = 5
        change_points = np.linspace(0, self.num_timesteps, num_changes+1, dtype=int)
        pattern = np.zeros(self.num_timesteps)
        
        for i in range(num_changes):
            pattern[change_points[i]:change_points[i+1]] = np.random.uniform(-0.5, 0.5) * self.acc_max
        return pattern
        
    def generate_pattern(self, pattern_type=None, return_name=False):
        if pattern_type is not None and pattern_type in self.patterns:
            pattern = self.patterns[pattern_type]()
            return (pattern, pattern_type) if return_name else pattern
        probabilities = {'uniform': 0.14,
                        'uniform_acc': 0.14,
                        'uniform_dec': 0.14,
                        'sinusoidal': 0.14,
                        'stop_and_go': 0.15,
                        'acc_dec_sequence': 0.15,
                        'random_changes': 0.14,}
        pattern_type = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        pattern = self.patterns[pattern_type]()
        
        return (pattern, pattern_type) if return_name else pattern

def run_simulation(env, agent, visualizer, pattern_generator):
    global rewards_history, global_step
    training_rewards, validation_rewards = [], []
    episode_count = 0

    while episode_count < num_episodes:
        # Training phase
        for _ in range(validation_freq):
            if episode_count >= num_episodes:
                break
            
            leader_actions, pattern_name = pattern_generator.generate_pattern(return_name=True)
            state = env.reset(leader_actions, pattern_name)
            score, episode_step = 0, 0
            visualize_episode = episode_count % visualization_freq == 0

            if visualize_episode:
                visualizer.reset_episode(episode_count)
            
            for _ in range(num_timesteps):
                if TABULAR_QL:
                    discrete_state = env.discretize_state(state, agent.state_bins)
                    action = agent.select_action(discrete_state)
                else:
                    action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward

                if episode_count % log_freq == 0:
                    wandb.log({
                        f"States/{episode_count+1} - EP": state[0],
                        f"States/{episode_count+1} - EV": state[1],
                        f"States/{episode_count+1} - ACC": state[2]
                    }, step = global_step)
                
                # Training updates
                if TABULAR_QL:
                    discrete_next_state = env.discretize_state(next_state, agent.state_bins)
                    agent.tab_update(discrete_state, action, reward, discrete_next_state, done)
                else:
                    agent.store_transition(state, action, reward, next_state, done)
                    if global_step % update_freq == 0:
                        agent.update()

                if visualize_episode:
                    visualizer.instant_rewards.append(reward)
                    visualizer.update(env)
                    visualizer.total_distance += env.leader_velocity * T
                    time.sleep(T)
                
                episode_step += 1
                global_step += 1
                if done:
                    break
                
                state = next_state

            training_rewards.append(score)
            agent.decay_epsilon()
            episode_count += 1

            wandb.log({
                "Training/Score": score,
                "Training/epsilon": agent.epsilon,
                "Episode": episode_count,
                "Training/Average Score": np.mean(training_rewards[-window_size:]),
                "Training/Steps": env.collision_step if env.collision_step is not None else episode_step,
            }, step=global_step)

            if visualize_episode:
                visualizer.total_reward = score
                visualizer.avg_reward = np.mean(training_rewards[-window_size:]) if len(training_rewards) > window_size else np.mean(training_rewards)

            print(f"Training Episode {episode_count + 1}, Steps: {env.collision_step if env.collision_step is not None else episode_step}, Epsilon: {agent.epsilon:.4f}, Score: {score:.4f}, Avg Score: {np.mean(training_rewards[-window_size:]):.4f}, Pattern: {env.pattern_name}")

        # Validation phase
        saved_epsilon = agent.epsilon
        agent.epsilon = 0       # Greedy policy for validation: no exploration
        validation_scores = []
        for _ in range(validation_episodes):
            leader_actions, pattern_name = pattern_generator.generate_pattern(return_name=True)
            state = env.reset(leader_actions, pattern_name)
            score = 0
            for _ in range(num_timesteps):
                if TABULAR_QL:
                    discrete_state = env.discretize_state(state, agent.state_bins)
                    action = agent.select_action(discrete_state)
                else:
                    action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break   
                state = next_state
            validation_scores.append(score)
        
        val_avg_score = np.mean(validation_scores)
        val_std_score = np.std(validation_scores)
        validation_rewards.append(val_avg_score)

        #agent.update_scheduler(val_avg_score)
        agent.epsilon = saved_epsilon
        
        wandb.log({
            "Validation/Average Score": val_avg_score,
            "Validation/Score Std": val_std_score,
        }, step=global_step)
        
        print(f"Validation after episode {episode_count}: Avg Score: {val_avg_score:.4f} ± {val_std_score:.4f}")

if __name__ == "__main__":
    SEED = 1492
    set_seeds(SEED)

    wandb.login()
    wandb_run = wandb.init(project="PlatoonControl", config=config, name=run_name)

    pattern_generator = LeaderPatternGenerator(num_timesteps, T, acc_max)
    leader_actions, pattern_name = pattern_generator.generate_pattern(return_name=True)
    rewards_history = []
    global_step = 0

    # Setup Environment
    env = EnvPlatoon(num_vehicles, vehicles_length, num_timesteps, T, h, tau, ep_max, ep_max_nominal,
                     ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold,
                     lambd, env_gamma, r, leader_min_speed, leader_max_speed, pattern_name, min_safe_distance, collision_penalty)
    env.seed(SEED)
    visualizer = PlatooningVisualizer(env)

    # Agent initialization (Q-Learning or DQN)
    if TABULAR_QL:
        agent = TabularQAgent(state_bins, action_size, env.u_min, env.u_max, lr, agent_gamma, epsilon, eps_decay, min_epsilon)
    else:
        agent = DQNAgent(state_size, hidden_size, action_size, u_min, u_max, lr, agent_gamma, soft_update_tau, epsilon, eps_decay,
                         min_epsilon, ep_max_nominal, ev_max_nominal, acc_min, acc_max, max_grad_norm, buffer_size, batch_size, device,
                         discrete_actions)
    
    simulation_done = threading.Event()
    sim_thread = threading.Thread(target = run_simulation, args = (env, agent, visualizer, pattern_generator))
    sim_thread.start()

    def update_task(task):
        if simulation_done.is_set():
            return task.done
        return task.cont

    visualizer.taskMgr.add(update_task, "UpdateTask")
    visualizer.accept("p", visualizer.toggle_pause)
    visualizer.run()

    sim_thread.join()
    visualizer.stop_visualizing()
    wandb.finish()
    wandb_run.finish()