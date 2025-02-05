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

The state space consists, at each timestep $k$, of three values: ${e_{p,i}^k, e_{v,i}^k, acc_i^k}$. The position gap-keeping error ($e_{p,i}^k$) and velocity error ($e_{v,i}^k$) provide the agent with direct information about the objectives to achieve, while the current acceleration ($acc_i^k$) allows the agent to consider system inertia in the decision-making process. For use in the neural network, these states are normalized with respect to their nominal maximum values, ensuring a uniform and well-conditioned input for learning.

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

- $a$ balances the importance of velocity error relative to the position gap-keeping error
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

The `EnvPlatoon` class implements the structure of the platoon environment. It contains numerous attributes that define the specific behavior of the system, imposing limits and constraints, and dimensioning the variables present in the equations that define each vehicle's dynamics.

The class includes a `reset` function, called at the beginning of each episode, which initializes a specific leader action pattern, resets rewards, and randomly samples an initial agent state within the defined intervals for each element of the state.

The `step` function applies first-order dynamics to the agent performing the state transition: acceleration, position gap-keeping error, and velocity error are calculated for the next timestep, and the reward is computed using the `compute_reward` function. While analyzing the reference paper, I noticed the absence of a mechanism to penalize the agent when its state would lead to a negative distance from the leader, which in reality would result in a collision. Therefore, in my implementation, I included an additional `collision_penalty` in the reward calculation that depends on the distance from the leader.

#### Environment attributes

|*Notation*|*Description*|*Value*|
|:-:|:-:|:-:|
|$T$|Interval for each timestep|0.1 s|
|$K$|Total timesteps in each episode|100|
|$N$|Number of vehicles|2|
|$\tau_i$|Driveline dynamics time constant|0.1 s|
|$h_i$|Time gap|1 s|
|$e_{p,max}$|Maximum initial position gap-keeping error|2 m|
|$e_{v,max}$|Maximum initial velocity error|1.5 m/s|
|$acc_{min}$|Minimum acceleration|-2.6 m/s^2|
|$acc_{max}$|Maximum acceleration|2.6 m/s^2|
|$u_{min}$|Minimum control input|-2.6 m/s^2|
|$u_{max}$|Maximum control input|2.6 m/s^2|
|$a$|Reward coefficient for the position gap-keeping error term|0.1|
|$b$|Reward coefficient for the velocity error term|0.1|
|$c$|Reward coefficient for the jerk term|0.1|
|$\lambda$|Reward scale|5e-3|
|$e_{p,max}^{nom}$|Nominal maximum position gap-keeping error|15 m|
|$e_{v,max}^{nom}$|Nominal maximum velocity error|10 m/s|
|$\varepsilon$|Reward threshold|-0.4483|

#### Leader patterns

While the reference paper uses real driving data extracted from the *Next Generation Simulation (NGSIM)* dataset, for the scope of this work I implemented a simple pattern generator that creates diverse but controlled scenarios for testing the agent's behavior. The `LeaderPatternGenerator` class creates seven different leader movement patterns that the agent must learn to follow:

- Uniform Motion (constant velocity)
- Uniformly Accelerated Motion with positive acceleration
- Uniformly Accelerated Motion with negative acceleration
- Sinusoidal Motion, simulating traffic flow patterns
- Stop-and-Go pattern typical of traffic situations
- Acceleration-Deceleration sequence
- Smooth random changes in acceleration

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/leader_patterns.png", width="45%" />
</p>
<p align="center"><i>Visualization of the different leader patterns.</i></p>

Each pattern is designed to test different aspects of the agent's learning capabilities and its ability to maintain proper distance in various driving scenarios.

### 3.2 - Agent development

The `DQNAgent` class implements the Deep Q-Learning agent. The underlying `DeepQNetwork` is a simple Multilayer Perceptron that has been tested with both one and two hidden layers. The implementation includes a configurable experience replay buffer.

Action selection is performed using an $\varepsilon$-greedy strategy, where $\varepsilon$ starts from an initial value and decays during training. The target network is updated through a soft update mechanism that allows for smoother learning, ensuring effective propagation of the main platooning task during training without distortions.

The `TabularQAgent` class follows the same logic for implementing the Tabular Q-Learning agent, but instead of having a target network, the Q-Table is simply defined as a NumPy array and is updated using the Bellman equation. The state and action spaces are discretized uniformly, creating a finite table of state-action values.

### 3.3 - Visualization

Since performance metrics such as the cumulative reward might not provide the best insight about the efficiency and quality of an agent's training, I decided to implement a visualization system using `Panda3D`. The renderer allows episode visualization, monitoring the agent's position relative to where it should be to maintain an optimal distance from the leader, marked by a red line and the text "DESIRED". This visualization tool helped in calibrating rewards and giving them appropriate weights relative to their magnitude. Below are two example frames from the visualization.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/panda2.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/panda3.png", width="45%" />
</p>
<p align="center"><i>Rendering of an episode using Panda3D.</i></p>

It is important to note that the implementation of a visualization system required a more realistic approach than one that just considers vehicles as points. Therefore, the equations were modified to account for vehicle lengths, transforming the simple point-to-point distance into a more realistic bumper-to-bumper distance. This distance is measured from the front bumper of the following vehicle to the rear bumper of the leading vehicle, which is essential for accurate collision detection and more realistic platooning behavior. This modification not only enhanced the visualization but also made the simulation more realistic by ensuring that the desired distances maintained by the agent consider the physical dimensions of the vehicles.

## 4 - Experimental Results

### Training with only one leader pattern: Uniform Motion

The training experiments were conducted in two main phases. In the first phase, I performed several training sessions with the leader movement pattern fixed at constant velocity (Uniform Motion). This allowed me to validate both the environment and the DQL learning method in a simplified setting, and to learn how modifications of each hyperparameter influence the training performance.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_onepattern_best.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_onepattern_best_2.png", width="45%" />
</p>
<p align="center"><i>Validation average score (over 100 episodes) of the four best DQL agents in a system with single leader pattern.</i></p>

### Training with different leader patterns

After reaching a satisfactory training performance in this basic scenario, I proceeded to the second phase where I introduced the seven different leader patterns in order to create a more challenging and diverse platooning task. Throughout this phase, a hyperparameter analysis was conducted in order to identify the optimal configuration for each learning method, compare the performance of different parameter settings, and evaluate the robustness of both approaches under varying initial conditions.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_2.png", width="45%" />
</p>
<p align="center"><i>Validation average score (over 100 episodes) of the three best DQL agents in a system with seven different leader patterns.</i></p>

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best2tab.png", width="45%" />
</p>
<p align="center"><i>Validation average score (over 100 episodes) of the two best Tabular Q-Learning agents compared with the best DQL agent.</i></p>

### Comparison with paper results

| |*Mean Episode Reward*|
|:-:|:-:|
|DDPG|-0.0680|
|FH-DDPG|-0.0736|
|HCFS|-0.0673|
|FH-DDPG-SS|-0.0600|
|**Tabular QL**|-0.1221|
|**Deep QL**|-0.1998|

### State evolution analysis

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state1000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state1000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state1000ev.png", width="33%" />
</p>
<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state5000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state5000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state5000ev.png", width="33%" />
</p>
<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state7000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state7000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state7000ev.png", width="33%" />
</p>
<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state9000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state9000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/best3dqn_state9000ev.png", width="33%" />
</p>
<p align="center"><i>State evolution of different training episodes (1000, 5000, 7000, 9000) of the three best DQL agents: $acc$ (left), $e_p$ (center), $e_v$ (right).</i></p>

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state1000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state1000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state1000ev.png", width="33%" />
</p>
<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state5000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state5000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state5000ev.png", width="33%" />
</p>
<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state8000acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state8000ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state8000ev.png", width="33%" />
</p>
<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state10400acc.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state10400ep.png", width="33%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/besttab_state10400ev.png", width="33%" />
</p>
<p align="center"><i>State evolution of different training episodes (1000, 5000, 8000, 10400) of the two best Tabular Q-Learning agents compared with the best DQL agent: $acc$ (left), $e_p$ (center), $e_v$ (right).</i></p>

### Hyperparameters analysis

#### Action space size

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_actionsize.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_actionsize_2.png", width="45%" />
</p>

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_actionsize.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_actionsize_2.png", width="45%" />
</p>

#### Batch size

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_batchsize.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_batchsize_2.png", width="45%" />
</p>

#### Experience Replay Buffer size

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_buffersize.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_buffersize_2.png", width="45%" />
</p>

#### Target Network update frequency

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_updatefreq_2.png", width="45%" />
</p>

#### Epsilon decay factor

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_epsdecay.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_epsdecay_2.png", width="45%" />
</p>

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_epsdecay.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_epsdecay_2.png", width="45%" />
</p>

#### Deep Q-Network hidden layers size

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/dqn_hiddensize.png", width="45%" />
</p>

---

#### State space size

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_statebins.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_statebins_2.png", width="45%" />
</p>

#### Discount factor

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_agentgamma10.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_agentgamma10_2.png", width="45%" />
</p>

#### Learning rate

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_lr.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/tab_lr_2.png", width="45%" />
</p>

## 5 - Conclusion

This Project Work presented a simplified implementation of an Autonomous Platoon Control environment using two different Q-Learning approaches. Comparisons between them and hyperparameters analysis were conducted, leading to interesting key findings about these Reinforcement Learning methods.

Future work could focus on extending the system to handle multiple following vehicles, implementing more sophisticated Reinforcement Learning algorithms such as PPO, and testing the system with real traffic data from the NGSIM dataset.
