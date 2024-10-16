import numpy as np
import torch
import threading
import time
import wandb

from env_platoon import EnvPlatoon
from visualizer import PlatooningVisualizer
from agent import DQNAgent, TabularQAgent


TABULAR_QL = False
num_episodes = 1000

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
min_safe_distance = 0.2
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

visualization_freq = 200
log_freq = 200

config = {
    "num_vehicles": num_vehicles,
    "vehicles_length": vehicles_length,
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
    "leader_min_speed": leader_min_speed,
    "leader_max_speed": leader_max_speed,
    "min_safe_distance": min_safe_distance,
    "collision_penalty": collision_penalty,
    "hidden_size": hidden_size,
    "action_size": action_size,
    "state_bins": state_bins,
    "lr": lr,
    "agent_gamma": agent_gamma,
    "soft_update_tau": soft_update_tau,
    "epsilon": epsilon,
    "eps_decay": eps_decay,
    "min_epsilon": min_epsilon,
    "buffer_size": buffer_size,
    "batch_size": batch_size,
    "update_freq": update_freq,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rewards_history = []
leader_actions = np.zeros(num_timesteps)
global_step = 0

def run_simulation(env, agent, visualizer):
    global rewards_history, global_step

    for episode in range(num_episodes):
        state = env.reset(leader_actions)
        visualizer.reset_episode(episode)
        visualize_episode = episode % visualization_freq == 0
        score = 0
        episode_step = 0

        if visualize_episode:
            visualizer.reset_episode(episode)
        
        for timestep in range(num_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward

            if episode % log_freq == 0:
                wandb.log({
                    f"State/{episode+1} - EP": state[0],
                    f"State/{episode+1} - EV": state[1],
                    f"State/{episode+1} - ACC": state[2]
                }, step=global_step)
            
            if TABULAR_QL:
                discrete_state = env.discretize_state(state, agent.state_bins)
                discrete_next_state = env.discretize_state(next_state, agent.state_bins)
                agent.update(discrete_state, action, reward, discrete_next_state, done)
            else:
                agent.store_transition(state, action, reward, next_state, done)
                if timestep % update_freq == 0:
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

        wandb.log({
            "Score": score,
            "epsilon decay": agent.epsilon,
            "Episode": episode,
            "Average Score": np.mean(rewards_history[-window_size:]),
            "Total Steps": env.collision_step if env.collision_step is not None else episode_step,
        }, step=global_step)
        
        rewards_history.append(score)
        agent.decay_epsilon()

        if visualize_episode:
            visualizer.total_reward = score
            if len(rewards_history) > window_size:
                visualizer.avg_reward = np.mean(rewards_history[-window_size:])
            else:
                visualizer.avg_reward = np.mean(rewards_history)
        
        if len(rewards_history) > window_size:
            avg_reward = np.mean(rewards_history[-window_size:])
            print(f"Episode {episode + 1}, Epsilon: {agent.epsilon:.4f}, Avg Reward (last {window_size}): {avg_reward:.4f}, Total Reward: {score:.4f}, Tot Steps: {env.collision_step if env.collision_step is not None else episode_step}")
        else:
            print(f"Episode {episode + 1}, Epsilon: {agent.epsilon:.4f}, Total Reward: {score:.4f}, Tot Steps: {env.collision_step if env.collision_step is not None else episode_step}")

if __name__ == "__main__":
    wandb.login()
    wandb.init(project="PlatoonControl", config=config)

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
    sim_thread = threading.Thread(target=run_simulation, args=(env, agent, visualizer))
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
