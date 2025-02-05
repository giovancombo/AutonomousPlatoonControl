# Autonomous Platoon Control with Q-Learning

This work was developed as a Project Work for the Autonomous Agents and Intelligent Robotics course, taught by Professor Giorgio Battistelli, as part of the Master's Degree in Artificial Intelligence at the University of Florence, Italy.

The main objective is to evaluate and compare different Reinforcement Learning algorithms for solving an *Autonomous Platoon Control* problem by partially reproducing, on a smaller scale, the experimental results obtained in the following reference paper:

> [Autonomous Platoon Control with Integrated Deep Reinforcement Learning and Dynamic Programming](https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/paper.pdf), Tong Liu, Lei Lei, Kan Zheng, Kuan Zhang; 2022.

## 1 - Introduction

*Autonomous Platoon Control* is a highly important task for the future of intelligent transportation systems. Through automated coordination of vehicles in platoon formation, it is possible to optimize traffic flow, reduce fuel consumption, and improve road safety. The main challenge is maintaining an optimal distance between a system of queued vehicles while they adapt to the leader's speed variations.

The setup of this problem follows exactly the one implemented in the reference paper, with the only simplification being the presence of a *single* agent vehicle and a single preceding vehicle, the leader. All vehicles follow a **first-order dynamics**:

$$\dot{p}_i(t) = v_i(t)$$
$$\dot{v}_i(t) = acc_i(t)$$
$$\dot{acc}_i(t) = -\frac{1}{\tau_i}acc_i(t) + \frac{1}{\tau_i}u_i(t)$$

where $\tau_i$ represents the time constant that models the delay in the vehicle's control system response. This parameter is crucial as it represents the system's inertia in responding to acceleration commands, directly affecting the control's stability and responsiveness.

To prevent divergences and unrealistic acceleration spikes that could compromise the agent's training, constraints are imposed on the possible values for the agent's acceleration and action:

$$acc_{min} \leq acc_i(t) \leq acc_{max}$$
$$u_{min} \leq u_i(t) \leq u_{max}$$

### Desired headway

The success of the platoon control task strongly depends on maintaining the correct distance between vehicles. In the reference paper, **headway** is defined as the *bumper-to-bumper* distance between two consecutive vehicles:

$$d_i(t) = p_{i-1}(t) - p_i(t) - L_{i-1}$$

where $L_{i-1}$ represents the length of the vehicle preceding vehicle $i$. For simplicity, we consider all vehicles to have the same length.

At any time instant $t$, each vehicle following the leader has its own desired headway from the preceding vehicle:

$$d_{r,i}(t) = r_i + h_iv_i(t)$$

where $r_i$ represents the safety distance that a stationary vehicle must maintain from the preceding one; and where $h_i$ represents the time constant given by the time it would take for the vehicle to reach (collide with) the preceding vehicle while maintaining a constant speed. This time-headway-based spacing policy significantly contributes to system stability: as speed increases, the desired safety distance increases proportionally, ensuring greater braking distance and therefore improved safety.

Optimal platoon control is achieved when each vehicle manages to adjust its motion dynamics to maintain the desired distance from the preceding vehicle over time. Consequently, *Platoon Control* can be easily transformed into a minimization problem by setting the objective as the minimization, by the agent, of two **error** values: one for achieving the correct distance from the preceding vehicle, and one for maintaining the correct speed to ensure this desired distance is not only reached but maintained over time.

$$e_{p,i}(t) = d_i(t) - d_{r,i}(t)$$
$$e_{v,i}(t) = v_{i-1}(t) - v_i(t)$$

### State and Action Space

The state space consists, at each timestep $k$, of three values: ${e_{p,i}^k, e_{v,i}^k, acc_i^k}$. The position error ($e_{p,i}^k$) and velocity error ($e_{v,i}^k$) provide the agent with direct information about the objectives to achieve, while the current acceleration ($acc_i^k$) allows the agent to consider system inertia in the decision-making process. For use in the neural network, these states are normalized with respect to their nominal maximum values, ensuring a uniform and well-conditioned input for learning.

The action space consists of a single value: $u_i^k \in [u_{min}, u_{max}]$.

### Dynamics

The system evolves according to two distinct discrete dynamic models for the leader and follower:

**Leader**: $$x_{0, k+1} = A_0x_{0,k} + B_0u_{0,k}$$

**Follower i**: $$x_{i, k+1} = A_ix_{i,k} + B_iu_{i,k} + C_iacc_{i-1,k}$$

For the leader, the evolution depends only on its current state and control input. For the follower, however, the evolution depends on its own state, its own control input, and the acceleration of the preceding vehicle. This dependence on the predecessor's acceleration allows the follower to anticipate speed variations of the preceding vehicle, thus making the system more stable.

### Reward system

A *Huber-like* reward function $R(S_i^k, u_i^k)$ is implemented. The choice of this particular reward function combines the advantages of both linear and quadratic functions: beyond a certain (negative) threshold of state transition reward, it switches from quadratic to absolute reward. This hybrid approach allows better handling of both large errors (through the linear component which is less sensitive to outliers) and small errors (through the quadratic component which provides a more precise gradient for fine optimization).

$$r_{abs} = -(|\frac{e_{p,i}^k}{e_{p,max}^{nom}}| + a|\frac{e_{v,i}^k}{e_{v,max}^{nom}}| + b|\frac{u_i^k}{u_{max}}| + c|\frac{j_i^k}{2acc_{max}/T}|)$$

$$r_{qua} = -\lambda{(e_{p,i}^k)^2 + a(e_{v,i}^k)^2 + b(u_i^k)^2 + c(j_i^kT)^2)}$$

The parameters $a$, $b$, and $c$ in the reward functions weight the relative importance of different terms:

- $a$ balances the importance of velocity error relative to position error
- $b$ penalizes overly aggressive control inputs, promoting smoother behavior
- $c$ penalizes sudden acceleration changes (jerk), contributing to driving comfort

$R(S_i^k, u_i^k) = r_{abs}$ if $r_{abs} < \epsilon$, otherwise $R(S_i^k, u_i^k) = r_{qua}$

Given the **expected cumulative reward** $J_{\pi_i} = E_{\pi_i}[\sum_{k=1}^K \gamma^{k-1}R(S_i^k, u_i^k)]$, the ultimate goal of the problem is to find a *policy* $\pi^*$ that **maximizes** $J_{\pi_i}$:

$$\pi^* = argmax_{\pi_i}(J_{\pi_i})$$

## 2 - Method

The reference paper proposes an integrated approach combining Deep Reinforcement Learning and Dynamic Programming, using an algorithm called FH-DDPG-SS. This method is based on DDPG (Deep Deterministic Policy Gradient) and is designed to handle a complex multi-agent system with multiple vehicles in platoon.

In this work, we implemented a simplified *single-agent* environment, where the agent must adjust its dynamics to match those of a single leading vehicle, which are set in advance. Therefore, we have a context with only two vehicles, one of which is the agent itself. The agent is trained using two different **Q-Learning** algorithms, whose performances will be compared:

- **Tabular Q-Learning**:  This represents the most "classical" approach to Reinforcement Learning, where the Q-function is explicitly represented as a table. While in DQL the state space is continuous, in Tabular Q-Learning both state space and action space are uniformly quantized. The Q-Table, having a value for each possible state-action pair, is initialized with random values in the range [-0.1, 0.1], and its update follows the *Bellman Equation*: $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$$, where $Q(s_t, a_t)$ is the current Q-value for the state-action pair; $\alpha$ is the learning rate; $r_t$ is the immediate reward; $\gamma$ is the discount factor; $\max_{a} Q(s_{t+1}, a)$ is the maximum possible Q-value in the next state; $[r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$ represents the TD error.

- **Deep Q-Learning (DQL)**: Deep Q-Learning extends classic Q-Learning by using a deep neural network to approximate the Q-function, making it possible to use a continuous state space. The implementation for this problem includes uniform quantization of the action space in the interval $[u_{min}, u_{max}]$; the use of an Experience Replay Buffer to store and sample state transitions; a Target Network to stabilize learning and propagate the original platooning task over time; and an ε-greedy policy to balance exploration and exploitation.

The implementation was developed in Python using PyTorch for DQL and NumPy for Tabular Q-Learning. Training was monitored through Weights & Biases.

## 3 - Code

### 3.1 - Environment development

`EnvPlatoon` è la classe in cui è implementata la struttura dell'environment di platooning. Si compone di numerosi attributi che definiscono il comportamento specifico del sistema, imponendo limiti e vincoli e dimensionando le variabili presenti nelle equazioni che definiscono la dinamica di ogni veicolo.

La classe si compone di una funzione `reset`, chiamata all'inizio di ogni episodio, che inizializza un particolare pattern d'azione del leader, resetta i reward ed estrae randomicamente uno stato iniziale dell'agente all'interno degli intervalli definiti per ciascun elemento dello stato.

La funzione `step` applica la dinamica di primo ordine all'agente effettuando la transizione di stato: sono calcolate accelerazione, position error and velocity error al successivo timestep, e contestualmente è calcolato il reward tramite la funzione `compute_reward`. Nel paper di riferimento è stata notata l'assenza di un meccanismo che consentisse di penalizzare l'agente nel caso in cui il proprio stato lo portasse a trovarsi a una distanza negativa dal leader, che nella realtà si tradurrebbe in un impatto. Pertanto, per la mia implementazione ho incluso nel calcolo del reward una ulteriore `collision_penalty` dipendente dalla distanza dal leader.

#### Environment attributes

|*Notation*|*Description*|*Value*|
|:-:|:-:|:-:|
|$T$|Interval for each timestep|0.1 s|
|$K$|Total timesteps in each episode|100|
|$N$|Number of vehicles|2|
|$\tau_i$|Driveline dynamics time constant|0.1 s|
|$h_i$|Time gap|1 s|
|$e_{p,max}$|Maximum initial gap-keeping error|2 m|
|$e_{v,max}$|Maximum initial velocity error|1.5 m/s|
|$acc_{min}$|Minimum acceleration|-2.6 m/s^2|
|$acc_{max}$|Maximum acceleration|2.6 m/s^2|
|$u_{min}$|Minimum control input|-2.6 m/s^2|
|$u_{max}$|Maximum control input|2.6 m/s^2|
|$a$|Reward coefficient for the gap-keeping error term|0.1|
|$b$|Reward coefficient for the velocity error term|0.1|
|$c$|Reward coefficient for the jerk term|0.1|
|$\lambda$|Reward scale|5e-3|
|$e_{p,max}^{nom}$|Nominal maximum gap-keeping error|15 m|
|$e_{v,max}^{nom}$|Nominal maximum velocity error|10 m/s|
|$\varepsilon$|Reward threshold|-0.4483|

#### Leader patterns

Il paper di riferimento non fa alcuna menzione riguardante il tipo di pattern seguito dal leader per determinare le risposte a ogni timestep dell'agente, poiché per l'addestramento si avvale di dati reali estratti dal dataset Next Generation Simulation (NGSIM).

Per lo scopo di questo lavoro, tuttavia, un semplice generatore di pattern è sufficiente. Viene infatti implementata la classe `LeaderPatternGenerator` che crea cinque differenti pattern di movimento del leader su cui l'agente deve imparare una policy:

- Moto Rettilineo Uniforme
- Moto Rettilineo Uniformemente accelerato con accelerazione positiva
- Moto Rettilineo Uniformemente accelerato con accelerazione negativa
- Moto Rettilineo Sinusoidale, a simulare l'andamento del traffico
- Moto Rettilineo con cambi randomici

### 3.2 - Agent development

La classe `DQNAgent` implementa l'agente che utilizza Deep Q-Learning per l'apprendimento. La `DeepQNetwork` su cui si fondano l'agente e la target network è un semplice Multilayer Perceptron che nell'esperienza è stato testato con uno o due hidden layers. Viene implementato un experience replay buffer di dimensione impostabile.

La selezione dell'azione viene eseguita tramite una trategia $\varepsilon$-*greedy*, con $\varepsilon$ che parte da un valore iniziale e decade durante il training. La target network viene aggiornata tramite un meccanismo di soft update che consente un apprendimento più dolce, garantendo una propagazione efficace durante il training del task principale di platooning senza distorsioni.

La classe `TabularQAgent` segue per l'implementazione dell'agente con Tabular Q-Learning la stessa logica dell'altro metodo, ma anziché avere una target network, la Q-Table è definita semplicemente come un NumPy array, e viene aggiornata tramite l'applicazione dell'equazione di Bellman.

#### Training hyperparameters




#### Leader patterns

Spiegazione dei vari attributi e metodi per la creazione dell'ambiente, confrontandoli con l'implementazione visibile nel paper di riferimento.
Breve spiegazione delle aggiunte che ho deciso di fare nel mio lavoro, legate particolarmente alla possibilità di visualizzare correttamente gli episodi, considerando ogni veicolo non più come un punto ma come un oggetto solido.

### 3.3 - Visualization
Spiegazione sommaria di come ho utilizzato Panda3D per visualizzare ogni episodio + piccola guida su come utilizzare e interpretare la visualizzazione.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/panda2.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/panda3.png", width="45%" />
</p>

<p align="center"><i>Rendering di episodi utilizzando Panda3D</i></p>

## 4 - Training

- Lista degli iperparametri dell'environment usati rimasti fissi
- Lista degli iperparametri valutati nel training

Iperparametri e range vari. Spiegazione della logica che ho adottato per raccogliere i vari risultati e fare i test.

1) Training con singolo pattern leader: moto rettilineo uniforme
2) Implementazione di 5 diversi pattern del leader
3) Implementazione Scheduler, gradient clipping, impostazione seed

## 5 - Results
Sequenza di plot con relativa spiegazione.

Risultati:
1) best run con moto rettilineo uniforme
2) best 3 run DQN
3) best DQN vs best TAB

Confronti:
- DQN action size
- DQN soft update
- DQN hidden size
- DQN epsilon decay
- DQN buffer size
- DQN batch size
- TAB gamma
- TAB learning rate
- TAB epsilon decay
- TAB action size
- TAB state bins

### Comparison with paper results

| |*Mean Episode Reward*|
|:-:|:-:|
|DDPG|-0.0680|
|FH-DDPG|-0.0736|
|HCFS|-0.0673|
|FH-DDPG-SS|-0.0600|
|**Tabular QL**|-0.1221|
|**Deep QL**|-0.1998|

## 6 - Conclusion
Recap generale dell'esperienza, menzionando i risultati ottenuti da ciascun metodo.
