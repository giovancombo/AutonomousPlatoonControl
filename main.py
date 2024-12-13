import numpy as np
import torch
import threading
import time
import wandb

from env_platoon import EnvPlatoon
from visualizer import PlatooningVisualizer
from agent import DQNAgent, TabularQAgent

TABULAR_QL = False          # True per usare il Q-Learning tabulare, False per usare il DQN
num_episodes = 5000
num_timesteps = 100         # Numero di step temporali per episodio

num_vehicles = 2
vehicles_length = 4         # Lunghezza fisica di ogni veicolo
T = 0.1                     # Intervallo di campionamento/controllo
tau = 0.1                   # Costante di tempo della dinamica del veicolo (risposta all'accelerazione)
h = 1.5                     # Time headway: tempo desiderato per raggiungere il veicolo precedente andando a velocità costante
r = 2                       # Standstill distance: distanza di sicurezza minima a veicolo fermo
min_safe_distance = 0.3     # Distanza minima assoluta per evitare collisioni in metri
collision_penalty = 0       # Penalità applicata in caso di collisione

ep_max = 2                  # Massimo errore di posizione ammissibile
ep_max_nominal = 15         # Valore di normalizzazione per l'errore di posizione
ev_max = 1.5                # Massimo errore di velocità ammissibile
ev_max_nominal = 10         # Valore di normalizzazione per l'errore di velocità
acc_min = -2.6              # Accelerazione minima possibile
acc_max = 2.6               # Accelerazione massima possibile

u_min = -2.6                # Input di controllo minimo
u_max = 2.6                 # Input di controllo massimo

a = 0.1                     # Peso per il termine di errore di velocità nel reward
b = 0.1                     # Peso per il termine di input di controllo nel reward
c = 0.2                     # Peso per il termine di jerk nel reward
reward_threshold = -0.4483  # Soglia per switchare tra reward assoluto e quadratico
lambd = 5e-3                # Fattore di scala per il reward quadratico
env_gamma = 1               # Fattore di sconto per il reward cumulativo (1 = no discount)

leader_min_speed = 50       # Velocità minima del leader in km/h
leader_max_speed = 50       # Velocità massima del leader in km/h

# Architettura della rete neurale
state_size = 3                                          # Dimensione dello spazio degli stati (ep, ev, acc)
hidden_size = [512, 256]                                # Dimensioni dei layer nascosti della rete

discrete_actions = True                                 # True (consigliato) per usare spazio delle azioni discreto invece che continuo
action_size = 1 if not discrete_actions else 10         # Dimensione dello spazio delle azioni
state_bins = (10, 10, 10)                               # Numero di bin per discretizzare ogni dimensione dello stato

lr = 5e-4                   # Learning rate per l'ottimizzazione
agent_gamma = 0.99          # Discount factor per i reward futuri
soft_update_tau = 0.01      # Coefficiente per soft update della target network

epsilon = 1.0               # Probabilità iniziale di esplorazione
eps_decay = 0.999           # Fattore di decadimento dell'epsilon
min_epsilon = 0.08          # Valore minimo di epsilon

buffer_size = 150000        # Dimensione massima del buffer di esperienza
batch_size = 256            # Dimensione del batch per l'addestramento
update_freq = 100           # Frequenza di aggiornamento della rete (ogni quanti step)

window_size = 100           # Finestra per il calcolo della media mobile delle performance
visualization_freq = 2000   # Frequenza di visualizzazione episodio (ogni quanti episodi)
log_freq = 200              # Frequenza di logging su wandb (ogni quanti episodi)

validation_freq = 100
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

leader_actions = np.zeros(num_timesteps)            # Pattern del leader (moto uniforme)
rewards_history = []                                # Storia dei reward per calcolare medie
global_step = 0                                     # Contatore globale degli step per logging

def run_simulation(env, agent, visualizer):
    global rewards_history, global_step
    
    training_rewards = []
    validation_rewards = []
    episode_count = 0

    while episode_count < num_episodes:
        # Training phase
        for _ in range(validation_freq):
            if episode_count >= num_episodes:
                break
                
            state = env.reset(leader_actions)
            visualize_episode = episode_count % visualization_freq == 0
            score = 0
            episode_step = 0
            
            if visualize_episode:
                visualizer.reset_episode(episode_count)
            
            for timestep in range(num_timesteps):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward

                if episode_count % log_freq == 0:
                    wandb.log({
                        f"States/{episode_count+1} - EP": state[0],
                        f"States/{episode_count+1} - EV": state[1],
                        f"States/{episode_count+1} - ACC": state[2]
                    }, step=global_step)
                
                # Training updates
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

            training_rewards.append(score)
            agent.decay_epsilon()
            
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

            print(f"Training Episode {episode_count + 1}, Epsilon: {agent.epsilon:.4f}, Score: {score:.4f}, Avg Score: {np.mean(training_rewards[-window_size:]):.4f}")
            episode_count += 1

        # Validation phase
        saved_epsilon = agent.epsilon
        agent.epsilon = 0  # Disabilita exploration durante validation
        
        validation_scores = []
        for _ in range(validation_episodes):
            state = env.reset(leader_actions)
            score = 0
            
            for timestep in range(num_timesteps):
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
        
        wandb.log({
            "Validation/Average Score": val_avg_score,
            "Validation/Score Std": val_std_score,
        }, step=global_step)
        
        print(f"Validation after episode {episode_count}: Avg Score: {val_avg_score:.4f} ± {val_std_score:.4f}")
        
        agent.epsilon = saved_epsilon

if __name__ == "__main__":
    wandb.login()
    wandb.init(project="PlatoonControl", config=config, name=run_name)

    # Setup Environment e Agente
    env = EnvPlatoon(num_vehicles, vehicles_length, num_timesteps, T, h, tau, ep_max, ep_max_nominal,
                     ev_max, ev_max_nominal, acc_min, acc_max, u_min, u_max, a, b, c, reward_threshold,
                     lambd, env_gamma, r, leader_min_speed, leader_max_speed, min_safe_distance, collision_penalty)
    visualizer = PlatooningVisualizer(env)

    # Inizializzazione agente (Q-Learning o DQN)
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

    # Avvio visualizzatore
    visualizer.taskMgr.add(update_task, "UpdateTask")
    visualizer.accept("p", visualizer.toggle_pause)
    visualizer.run()

    # Cleanup
    sim_thread.join()
    visualizer.stop_visualizing()
    wandb.finish()