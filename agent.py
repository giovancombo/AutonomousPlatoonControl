import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, hidden_size, action_size, u_min, u_max, lr, gamma, 
                 soft_update_tau, epsilon, eps_decay, min_epsilon, ep_max_nominal, 
                 ev_max_nominal, acc_min, acc_max, buffer_size, batch_size, device='cpu',
                 discrete_actions=False):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.u_min, self.u_max = u_min, u_max
        self.lr = lr
        self.gamma = gamma
        self.soft_update_tau = soft_update_tau
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_epsilon = min_epsilon
        self.ep_max_nominal = ep_max_nominal
        self.ev_max_nominal = ev_max_nominal
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.device = device
        self.discrete_actions = discrete_actions

        self.q_network = DeepQNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_q_network = DeepQNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def normalize_state(self, state):
        """Normalizza lo stato per l'input alla rete"""
        ep, ev, acc = state
        norm_ep = ep / self.ep_max_nominal
        norm_ev = ev / self.ev_max_nominal
        norm_acc = (acc - self.acc_min) / (self.acc_max - self.acc_min)
        return np.array([norm_ep, norm_ev, norm_acc])

    def select_action(self, state):
        """Seleziona un'azione usando la strategia ε-greedy"""
        if np.random.random() < self.epsilon:
            # Exploration
            action = np.random.uniform(self.u_min, self.u_max)
            if self.discrete_actions:
                discrete_actions = np.linspace(self.u_min, self.u_max, self.action_size)
                action = discrete_actions[np.abs(discrete_actions - action).argmin()]
        else:
            # Policy Exploitation
            with torch.no_grad():
                normalized_state = self.normalize_state(state)
                state_tensor = torch.tensor(normalized_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                if self.discrete_actions:
                    action_index = q_values.argmax().item()
                    action = self.u_min + (action_index / (self.action_size - 1)) * (self.u_max - self.u_min)
                else:
                    action = np.clip(q_values.item(), self.u_min, self.u_max)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Memorizza una transizione nel replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """Aggiorna la rete usando un batch di esperienze"""
        if len(self.memory) < self.batch_size:
            return

        # Campionamento batch di transizioni
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Normalizzazione stati
        normalized_states = [self.normalize_state(state) for state in states]
        normalized_next_states = [self.normalize_state(state) for state in next_states]

        states = torch.tensor(normalized_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(normalized_next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Calcolo Q-values correnti
        current_q_values = self.q_network(states)

        if self.discrete_actions:
            actions_index = ((actions - self.u_min) / (self.u_max - self.u_min) * (self.action_size - 1)).long()
            current_q_values = current_q_values.gather(1, actions_index)
            with torch.no_grad():
                next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
        else:
            current_q_values = current_q_values.squeeze(1)
            with torch.no_grad():
                next_actions = self.q_network(next_states).detach()
                next_q_values = self.target_q_network(next_states).gather(1, next_actions.max(1)[1].unsqueeze(1))

        # Calcola target Q-values usando l'equazione di Bellman
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update_target_network()

    def _soft_update_target_network(self):
        """Aggiorna la target network usando soft update"""
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.soft_update_tau * local_param.data + 
                                  (1.0 - self.soft_update_tau) * target_param.data)

    def decay_epsilon(self):
        """Riduce epsilon secondo il decay rate"""
        self.epsilon = max(self.epsilon * self.eps_decay, self.min_epsilon)

    def reset(self):
        """Resetta l'agente al suo stato iniziale"""
        self.memory.clear()
        self.epsilon = self.initial_epsilon


class TabularQAgent:
    def __init__(self, state_bins, num_actions, u_min, u_max, lr, gamma, epsilon, 
                 eps_decay, min_epsilon):
        self.state_bins = state_bins
        self.num_actions = num_actions
        self.u_min = u_min
        self.u_max = u_max
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_epsilon = min_epsilon

        self.discrete_actions = np.linspace(u_min, u_max, num_actions)
        
        # Q-table inizializzata a zero
        self.q_table = np.zeros(self.state_bins + (self.num_actions,))

    def select_action(self, state):
        """Seleziona un'azione usando la strategia ε-greedy"""
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(0, self.num_actions)
        else:
            action_index = np.argmax(self.q_table[state])
        return self.discrete_actions[action_index]

    def update(self, state, action, reward, next_state, done):
        """Aggiorna la Q-table usando l'equazione di Bellman"""
        action_index = np.abs(self.discrete_actions - action).argmin()
        current_q = self.q_table[state + (action_index,)]

        if not done:
            max_next_q = np.max(self.q_table[next_state])
            new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        else:
            new_q = current_q + self.lr * (reward - current_q)
        
        self.q_table[state + (action_index,)] = new_q

    def decay_epsilon(self):
        """Riduce epsilon secondo il decay rate"""
        self.epsilon = max(self.epsilon * self.eps_decay, self.min_epsilon)

    def reset(self):
        """Resetta l'agente al suo stato iniziale"""
        self.epsilon = self.initial_epsilon
        self.q_table = np.zeros(self.state_bins + (self.num_actions,))

    def get_q_value(self, state, action):
        """Restituisce il Q-value per una coppia stato-azione"""
        action_index = np.abs(self.discrete_actions - action).argmin()
        return self.q_table[state + (action_index,)]

    def get_best_action(self, state):
        """Restituisce l'azione con il massimo Q-value per uno stato"""
        action_index = np.argmax(self.q_table[state])
        return self.discrete_actions[action_index]

    def save_q_table(self, filename):
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.load(filename)