# Autonomous Platoon Control with Reinforcement Learning

Questo lavoro nasce come Project Work del corso di Autonomous Agents and Intelligent Robotics, tenuto dal professor Giorgio Battistelli, nell'ambito del corso di laurea magistrale in Intelligenza Artificiale all'Università degli Studi di Firenze, Italia.

Obiettivo principale è la valutazione e il confronto di algoritmi di Reinforcement Learning per la risoluzione di un problema di *Autonomous Platoon Control*, conducendo la parziale ricostruzione, su piccola scala, dei risultati sperimentali ottenuti dal seguente paper di riferimento:

> [Autonomous Platoon Control with Integrated Deep Reinforcement Learning and Dynamic Programming](https://github.com/giovancombo/AutonomousPlatoonControl/blob/main/paper.pdf), Tong Liu, Lei Lei, Kan Zheng, Kuan Zhang; 2022.

## 1 - Introduction

L'*Autonomous Platoon Control* è un task di elevata importanza per il futuro dei sistemi di trasporto intelligenti. Attraverso la coordinazione automatizzata di veicoli in plotone, è possibile ottimizzare il flusso del traffico, ridurre il consumo di carburante e migliorare la sicurezza stradale. La sfida principale consiste nel mantenere una distanza ottimale tra un sistema di veicoli in coda mentre questi si adattano alle variazioni di velocità del leader.

Il setting del seguente problema è stato ripreso totalmente da quello implementato nel paper di riferimento, con l'unica semplificazione data dalla presenza di un *singolo* veicolo agente, e un singolo altro veicolo che lo precede, il leader. Tutti i veicoli seguono una **dinamica del primo ordine**:

$$\dot{p}_i(t) = v_i(t)$$
$$\dot{v}_i(t) = acc_i(t)$$
$$\dot{acc}_i(t) = -\frac{1}{\tau_i}acc_i(t) + \frac{1}{\tau_i}u_i(t)$$

dove $\tau_i$ rappresenta la costante di tempo che modella il ritardo nella risposta del sistema di controllo del veicolo. Questo parametro è cruciale in quanto rappresenta l'inerzia del sistema nel rispondere ai comandi di accelerazione, influenzando direttamente la stabilità e la reattività del controllo.

Per prevenire divergenze ed esplosioni irrealistiche di accelerazioni che potrebbero compromettere l'addestramento dell'agente, vengono imposti dei vincoli ai valori assumibili dall'accelerazione dell'agente e dall'azione:

$$acc_{min} \leq acc_i(t) \leq acc_{max}$$
$$u_{min} \leq u_i(t) \leq u_{max}$$

#### Desired headway

Il successo del task di controllo del plotone dipende fortemente dal mantenimento di una corretta distanza tra i veicoli. Nel paper di riferimento, si definisce **headway** la distanza *bumper-to-bumper* tra due veicoli consecutivi:

$$d_i(t) = p_{i-1}(t) - p_i(t) - L_{i-1}$$

dove $L_{i-1}$ rappresenta la lunghezza del veicolo che precede il veicolo $i$. Per semplicità, consideriamo veicoli aventi tutti la medesima lunghezza.

In qualsiasi istante di tempo $t$, ogni veicolo che segue il leader possiede una propria headway desiderata dal veicolo che lo precede:

$$d_{r,i}(t) = r_i + h_iv_i(t)$$

dove $r_i$ rappresenta la distanza di sicurezza che un veicolo, da fermo, deve mantenere dal precedente; e dove $h_i$ rappresenta la costante di tempo data dal tempo che il veicolo impiegherebbe a raggiungere (collidere con) il veicolo precedente mantenendo una velocità costante. Questo approccio di spacing policy basato sul time headway contribuisce significativamente alla stabilità del sistema: al crescere della velocità, aumenta proporzionalmente anche la distanza di sicurezza desiderata, garantendo maggiore spazio di frenata e quindi maggiore sicurezza.

Il controllo ottimale del plotone si raggiunge nel momento in cui ogni veicolo riesce ad adeguare la propria dinamica di moto in modo da mantenere nel tempo la distanza desiderata dal veicolo che lo precede. Di conseguenza, il *Platoon Control* può essere facilmente reso un problema di minimizzazione imponendo come obiettivo la minimizzazione, da parte dell'agente, di due valori di **errore**, uno sulla distanza corretta da raggiungere rispetto al veicolo precedente, e uno sulla velocità corretta da mantenere affinché tale distanza desiderata non sia solo raggiunta, ma mantenuta nel tempo.

$$e_{p,i}(t) = d_i(t) - d_{r,i}(t)$$
$$e_{v,i}(t) = v_{i-1}(t) - v_i(t)$$

#### State and Action Space

Lo state space si compone, ad ogni timestep $k$, di tre valori: $\{e_{p,i}^k, e_{v,i}^k, acc_i^k\}$. L'errore di posizione ($e_{p,i}^k$) e l'errore di velocità ($e_{v,i}^k$) forniscono all'agente informazioni dirette sugli obiettivi da raggiungere, mentre l'accelerazione corrente ($acc_i^k$) permette all'agente di considerare l'inerzia del sistema nel processo decisionale. Per l'utilizzo nella rete neurale, questi stati vengono normalizzati rispetto ai loro valori massimi nominali, garantendo un input uniforme e ben condizionato per l'apprendimento.

L'action space si compone di un unico valore: $u_i^k \in [u_{min}, u_{max}]$.

#### Dynamics

Il sistema evolve secondo due modelli dinamici discreti distinti per il leader e il follower:

**Leader**: $$x_{0, k+1} = A_0x_{0,k} + B_0u_{0,k}$$

**Follower i**: $$x_{i, k+1} = A_ix_{i,k} + B_iu_{i,k} + C_iacc_{i-1,k}$$

Per il leader, l'evoluzione dipende solo dal suo stato attuale e dall'input di controllo. Per il follower, invece, l'evoluzione dipende dal proprio stato, dal proprio input di controllo e dall'accelerazione del veicolo che lo precede. Questa dipendenza dall'accelerazione del predecessore permette al follower di anticipare le variazioni di velocità del veicolo che lo precede, e rendere così più stabile il sistema.

#### Reward system

Viene implementata una funzione di reward *Huber-like* $R(S_i^k, u_i^k)$. La scelta di questa particolare funzione di reward combina i vantaggi di una funzione lineare e di una quadratica: oltre una certa soglia (negativa) di reward di transizione di stato, si passa dal reward quadratico a quello assoluto. Questo approccio ibrido permette di gestire meglio sia gli errori grandi (attraverso la componente lineare che è meno sensibile agli outliers) che quelli piccoli (attraverso la componente quadratica che fornisce un gradiente più preciso per l'ottimizzazione fine).

$$r_{abs} = -(|\frac{e_{p,i}^k}{e_{p,max}^{nom}}| + a|\frac{e_{v,i}^k}{e_{v,max}^{nom}}| + b|\frac{u_i^k}{u_{max}}| + c|\frac{j_i^k}{2acc_{max}/T}|)$$

$$r_{qua} = -\lambda{(e_{p,i}^k)^2 + a(e_{v,i}^k)^2 + b(u_i^k)^2 + c(j_i^kT)^2)}$$

I parametri $a$, $b$ e $c$ nelle funzioni di reward pesano l'importanza relativa dei diversi termini:

- $a$ bilancia l'importanza dell'errore di velocità rispetto all'errore di posizione
- $b$ penalizza l'utilizzo di input di controllo troppo aggressivi, promuovendo un comportamento più "smooth"
- $c$ penalizza variazioni brusche di accelerazione (jerk), contribuendo al comfort di guida

$R(S_i^k, u_i^k) = r_{abs}$ se $r_{abs} < \epsilon$, altrimenti $R(S_i^k, u_i^k) = r_{qua}$

Definito l'**expected cumulative reward** $J_{\pi_i} = E_{\pi_i}[\sum_{k=1}^K \gamma^{k-1}R(S_i^k, u_i^k)]$, l'obiettivo ultimo del problema è quello di trovare una *policy* $\pi^*$ che **massimizza** $J_{\pi_i}$:

$$\pi^* = argmax_{\pi_i}J_{\pi_i}$$

## 2 - Method

Il paper di riferimento propone un approccio integrato che combina Deep Reinforcement Learning e Dynamic Programming, utilizzando un algoritmo chiamato FH-DDPG-SS. Questo metodo si basa su DDPG (Deep Deterministic Policy Gradient) ed è progettato per gestire un sistema multi-agente complesso con numerosi veicoli in plotone.

In questo lavoro, è stato implementato un setting semplificato *a singolo agente*, nel quale l'agente deve adeguare la propria dinamica di modo a quella, imposta a priori, di un solo altro veicolo leader. Pertanto, ci si trova in un contesto dato da due soli veicoli, di cui uno è l'agente stesso. L'agente viene addestrato utilizzando due differenti algoritmi di **Q-Learning**, le cui performance saranno confrontate:

- **Tabular Q-Learning**: Il Q-Learning tabellare rappresenta l'approccio più "classico" al Reinforcement Learning, in cui la Q-function viene rappresentata esplicitamente come una tabella. Mentre nel DQL lo state space è continuo, nel Tabular Q-Learning lo state space viene quantizzato uniformemente, così come l'action space. La Q-Table, avente un valore per ogni coppia stato-azione possibile, è inizializzata con valori casuali nell'intervallo [-0.1, 0.1], e il suo aggiornamento segue la classica equazione di Bellman:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$$

dove $Q(s_t, a_t)$ è il valore Q corrente per la coppia stato-azione; $\alpha$ è il learning rate; $r_t$ è il reward immediato; $\gamma$ è il discount factor; $\max_{a} Q(s_{t+1}, a)$ è il massimo valore Q possibile nello stato successivo; $[r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$ rappresenta il TD error.

- **Deep Q-Learning (DQL)**: Il Deep Q-Learning estende il classico Q-Learning utilizzando una rete neurale profonda per approssimare la Q-function, rendendo possibile l'utilizzo di uno state space continuo. L'implementazione per questo problema prevede la quantizzazione uniforme dell'action space nell'intervallo $[u_{min}, u_{max}]$; l'utilizzo di un Experience Replay Buffer per memorizzare e campionare le transizioni di stato; una Target Network per stabilizzare l'apprendimento e propagare nel tempo il task originario di platooning; una ε-greedy policy per il bilanciamento tra exploration ed exploitation.

L'implementazione è stata realizzata in Python utilizzando PyTorch per il DQL e NumPy per il Q-Learning tabellare. Il training è stato monitorato attraverso Weights & Biases.

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
|**Deep QL**|-0.1639|

## 6 - Conclusion
Recap generale dell'esperienza, menzionando i risultati ottenuti da ciascun metodo.
