# Autonomous Platoon Control with Reinforcement Learning

Questo progetto nasce come Project Work relativo al corso di Autonomous Agents and Intelligent Robotics, tenuto dal professor Giorgio Battistelli, nell'ambito del corso di laurea magistrale in Intelligenza Artificiale all'Università degli Studi di Firenze, Italia.

Obiettivo principale è la valutazione e il confronto di algoritmi di Reinforcement Learning per la risoluzione di un problema di *Autonomous Platoon Control*, conducendo la parziale ricostruzione, su piccola scala, dei risultati sperimentali ottenuti dal seguente paper di riferimento:

> [Autonomous Platoon Control with Integrated Deep Reinforcement Learning and Dynamic Programming](https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/paper.pdf), Tong Liu, Lei Lei, Kan Zheng, Kuan Zhang; 2022.

Il setting, semplificato rispetto a quanto descritto nel paper, prevede la creazione di un environment *a singolo agente*, nel quale l'agente deve adeguare la propria dinamica di modo a quella, imposta a priori, di un solo altro veicolo leader. Pertanto, ci si trova in un contesto dato da due soli veicoli, di cui uno è l'agente stesso.

L'agente viene addestrato utilizzando due differenti algoritmi di **Q-Learning**:
- **Deep Q-Learning**: prevede la discretizzazione dell'*action space*, ma non lo *state space*;
- **Q-Learning tabellare**: prevede la discretizzazione sia dell'*action space*, sia dello *state space*, in quanto utilizza una *Q-Table*.

Le performance dell'agente per ogni run e metodo sono state misurate e salvate utilizzando il supporto di Weights & Biases.

Data la semplicità strutturale dell'environment, è stato implementato un semplice visualizzatore che utilizza Panda3D per effettuare un rendering indicativo degli episodi.


## 1 - Introduction

Il setting del seguente problema di *Autonomous Platoon Control* è stato ripreso totalmente da quello implementato nel paper di riferimento, con l'unica semplificazione data dalla presenza di un *singolo* veicolo agente, e un singolo altro veicolo che lo precede, il leader. Tutti i veicoli seguono una **dinamica del primo ordine**:

$$\dot{p}_i(t) = v_i(t)$$
$$\dot{v}_i(t) = acc_i(t)$$
$$\dot{acc}_i(t) = -\frac{1}{\tau_i}acc_i(t) + \frac{1}{\tau_i}u_i(t)$$

Con $\tau_i$ = costante di tempo dovuta al transitorio presente nel controllo.

Per prevenire divergenze, esplosioni irrealistiche di accelerazioni e quindi la compromissione dell'addestramento dell'agente, vengono imposti dei vincoli ai valori assumibili dall'accelerazione dell'agente e dall'azione:

$$acc_{min} \leq acc_i(t) \leq acc_{max}$$
$$u_{min} \leq u_i(t) \leq u_{max}$$

#### Desired headway

Il successo del task di controllo del plotone dipende fortemente dal mantenimento di una corretta distanza tra i veicoli. Nel paper di riferimento, si definisce **headway** la distanza *bumper-to-bumper* tra due veicoli consecutivi:

$$d_i(t) = p_{i-1}(t) - p_i(t) - L_{i-1}$$

Con $L_{i-1}$ = lunghezza del veicolo che precede il veicolo $i$. Per semplicità, consideriamo veicoli aventi tutti la medesima lunghezza.

In qualsiasi istante di tempo $t$, ogni veicolo che segue il leader possiede una propria **headway** (= distanza) **desiderata** dal veicolo che lo precede:

$$d_{r,i}(t) = r_i + h_iv_i(t)$$

Con $r_i$ = distanza di sicurezza che un veicolo, da fermo, deve mantenere dal precedente; e con $h_i$ = costante di tempo data dal tempo che il veicolo impiegherebbe a raggiungere (collidere con) il veicolo precedente mantenendo una velocità costante.

Il controllo ottimale del plotone si raggiunge nel momento in cui ogni veicolo riesce ad adeguare la propria dinamica di moto in modo da mantenere nel tempo la distanza desiderata dal veicolo che lo precede. Di conseguenza, il *Platoon Control* può essere facilmente reso un problema di minimizzazione imponendo come obiettivo la minimizzazione, da parte dell'agente, di due valori di **errore**, uno sulla distanza corretta da raggiungere rispetto al veicolo precedente, e uno sulla velocità corretta da mantenere affinché tale distanza desiderata non sia solo raggiunta, ma mantenuta nel tempo.

$$e_{p,i}(t) = d_i(t) - d_{r,i}(t)$$
$$e_{v,i}(t) = v_{i-1}(t) - v_i(t)$$

#### State and Action Space

Lo *state space* si compone, ad ogni timestep $k$, di tre valori: $`\{e_{p,i}^k, e_{v,i}^k, acc_i^k\}`$

L'*action space* si compone di un unico valore: $u_i^k \in [u_{min}, u_{max}]$

#### Dynamics

**Leader**: $$x_{0, k+1} = A_0x_{0,k} + B_0u_{0,k}$$

**Follower i**: $$x_{i, k+1} = A_ix_{i,k} + B_iu_{i,k} + C_iacc_{i-1,k}$$

#### Reward system

Viene implementata una funzione di reward *Huber-like* $R(S_i^k, u_i^k)$, per la quale oltre una certa soglia di reward di transizione di stato negativo, si passa dal reward quadratico a quello assoluto.

$$r_{abs} = -(|\frac{e_{p,i}^k}{e_{p,max}^{nom}}| + a|\frac{e_{v,i}^k}{e_{v,max}^{nom}}| + b|\frac{u_i^k}{u_{max}}| + c|\frac{j_i^k}{2acc_{max}/T}|)$$

$$r_{qua} = -\lambda{(e_{p,i}^k)^2 + a(e_{v,i}^k)^2 + b(u_i^k)^2 + c(j_i^kT)^2)}$$

$R(S_i^k, u_i^k) = r_{abs}$ se $r_{abs} < \epsilon$, altrimenti $R(S_i^k, u_i^k) = r_{qua}$

Definito l'**expected cumulative reward** $J_{\pi_i} = E_{\pi_i}[\sum_{k=1}^K \gamma^{k-1}R(S_i^k, u_i^k)]$, l'obiettivo ultimo del problema è quello di trovare una *policy* $\pi^*$ che **massimizza** $J_{\pi_i}$:

$$\pi^* = argmax_{\pi_i}J_{\pi_i}$$

## 2 - Method
Breve introduzione teorica dei metodi di Deep Q-Learning o Q-Learning Tabulare. Aspettative generali di funzionamento dei due metodi.
Spiegazione sommaria della creazione dell'Agente.

## 3 - Code

### 3.1 - Environment development
Spiegazione dei vari attributi e metodi per la creazione dell'ambiente, confrontandoli con l'implementazione visibile nel paper di riferimento.
Breve spiegazione delle aggiunte che ho deciso di fare nel mio lavoro, legate particolarmente alla possibilità di visualizzare correttamente gli episodi, considerando ogni veicolo non più come un punto ma come un oggetto solido.

### 3.2 - Visualization
Spiegazione sommaria di come ho utilizzato Panda3D per visualizzare ogni episodio + piccola guida su come utilizzare e interpretare la visualizzazione.

<p float="left", align="center">
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/panda2.png", width="45%" />
  <img src="https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/images/panda3.png", width="45%" />
</p>

<p align="center"><i>Rendering di episodi utilizzando Panda3D</i></p>

## 4 - Training
Setup hardware e software; Iperparametri e range vari. Spiegazione della logica che ho adottato per raccogliere i vari risultati e fare i test.

## 5 - Results
Sequenza di plot con relativa spiegazione.

### Comparison with paper results

| |*Mean Episode Reward*|
|:-:|:-:|
|DDPG|-0.0680|
|FH-DDPG|-0.0736|
|HCFS|-0.0673|
|FH-DDPG-SS|-0.0600|
|**QL Tabellare**|?|
|**Deep QL**|-0.9466|

## 6 - Conclusion
Recap generale dell'esperienza, menzionando i risultati ottenuti da ciascun metodo.
