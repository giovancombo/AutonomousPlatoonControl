import numpy as np
import torch
import threading
import time

from env_platoon import EnvPlatoon
from visualizer import PlatooningVisualizer
from agent import DQNAgent, TabularQAgent


TABULAR_QL = False
num_episodes = 3000

num_vehicles = 2
vehicles_length = 4
num_timesteps = 100
T = 0.1
tau = 0.1
h = 1.5
ep_max = 2
ep_max_nominal = 15
ev_max = 1.5
ev_max_nominal = 10
acc_min = -2.6
acc_max = 2.6
u_min = -2.6
u_max = 2.6
a = 0.1
b = 0.1
c = 0.2
reward_threshold = -0.4483  # Paper: -0.4483
lambd = 5e-3
env_gamma = 1               # Non utile per il Q-Learning
r = 2
leader_min_speed = 50       # km/h
leader_max_speed = 50
min_safe_distance = 0.3
collision_penalty = -2

# Parametri dell'agente
discrete_actions = True

state_size = 3
hidden_size = [256, 128]
action_size = 1 if not discrete_actions else 200
state_bins = (50, 50, 50)
lr = 0.005
agent_gamma = 0.99
soft_update_tau = 0.01
epsilon = 1.0
eps_decay = 0.9999
min_epsilon = 0.01
buffer_size = 100000
batch_size = 128

window_size = 100
update_freq = 4

visualization_freq = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_simulation(env, agent, visualizer):
    global rewards_history
    for episode in range(num_episodes):
        state = env.reset(leader_actions)
        visualizer.reset_episode(episode)
        total_reward = 0
        steps = 0

        visualize_episode = episode % visualization_freq == 0
        if visualize_episode:
            visualizer.reset_episode(episode)
        
        for step in range(num_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if TABULAR_QL:
                discrete_state = env.discretize_state(state, agent.state_bins)
                discrete_next_state = env.discretize_state(next_state, agent.state_bins)
                agent.update(discrete_state, action, reward, discrete_next_state, done)
            else:
                agent.store_transition(state, action, reward, next_state, done)
                if step % update_freq == 0:
                    agent.update()

            if visualize_episode:
                visualizer.instant_rewards.append(reward)
                visualizer.update(env)
                visualizer.total_distance += env.leader_velocity * T
                time.sleep(T)
            steps += 1
            
            if done:
                print(f"next_ep: {state[0]}")
                print(f"next_ev: {state[1]}")
                print(f"next_acc: {state[2]}")
                break
            
            state = next_state
        
        rewards_history.append(total_reward)
        agent.decay_epsilon()

        if visualize_episode:
            visualizer.total_reward = total_reward
            if len(rewards_history) > window_size:
                visualizer.avg_reward = np.mean(rewards_history[-window_size:])
            else:
                visualizer.avg_reward = np.mean(rewards_history)
        
        if len(rewards_history) > window_size:
            avg_reward = np.mean(rewards_history[-window_size:])
            print(f"Episode {episode + 1}, Total Reward: {total_reward:.4f}, Avg Reward (last {window_size}): {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}, Tot Steps: {env.collision_step if env.collision_step is not None else steps}")
        else:
            print(f"Episode {episode + 1}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}, Tot Steps: {env.collision_step if env.collision_step is not None else steps}")

rewards_history = []
leader_actions = np.zeros(num_timesteps)

if __name__ == "__main__":
    env = EnvPlatoon(num_vehicles, vehicles_length, num_timesteps, T, h, tau, ep_max, ep_max_nominal,
                        ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold,
                        lambd, env_gamma, r, leader_min_speed, leader_max_speed, min_safe_distance, collision_penalty)
    visualizer = PlatooningVisualizer(env)

    if TABULAR_QL:
        agent = TabularQAgent(state_bins, action_size, env.u_min, env.u_max, lr, agent_gamma, epsilon, eps_decay, min_epsilon)
    else:
        agent = DQNAgent(state_size, hidden_size, action_size, u_min, u_max, lr, agent_gamma, soft_update_tau, epsilon, eps_decay,
                        min_epsilon, ep_max_nominal, ev_max_nominal, acc_min, acc_max, buffer_size, batch_size, device,
                        discrete_actions)
    
    simulation_done = threading.Event()
    sim_thread = threading.Thread(target = run_simulation, args = (env, agent, visualizer))
    sim_thread.start()

    def update_task(task):
        if simulation_done.is_set():
            return task.done
        return task.cont

    visualizer.taskMgr.add(update_task, "UpdateTask")
    visualizer.accept("p", visualizer.toggle_pause)  # Premere 'p' per mettere in pausa/riprendere
    visualizer.run()

    # Aspetta che la simulazione finisca
    sim_thread.join()
    visualizer.stop_visualizing()
