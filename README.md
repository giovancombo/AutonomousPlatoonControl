# Autonomous Platoon Control with Reinforcement Learning

## 1 - Introduction
Introduzione del problema: replicare il funzionamento del paper nel quale si effettua Autonomous Platoon Control, ma utilizzando Deep Q-Learning o Q-Learning Tabulare, e confrontarne le performance. Introduzione generale del lavoro, consistente nella ricostruzione dell'ambiente da zero, e una visualizzazione sommaria degli episodi, oltre al tracking con wandb.

## 2 - Method
Breve introduzione teorica dei metodi di Deep Q-Learning o Q-Learning Tabulare. Aspettative generali di funzionamento dei due metodi.
Spiegazione sommaria della creazione dell'Agente.

## 3 - Code

### 3.1 - Environment development
Spiegazione dei vari attributi e metodi per la creazione dell'ambiente, confrontandoli con l'implementazione visibile nel paper di riferimento.
Breve spiegazione delle aggiunte che ho deciso di fare nel mio lavoro, legate particolarmente alla possibilità di visualizzare correttamente gli episodi, considerando ogni veicolo non più come un punto ma come un oggetto solido.

### 3.2 - Visualization
Spiegazione sommaria di come ho utilizzato Panda3D per visualizzare ogni episodio + piccola guida su come utilizzare e interpretare la visualizzazione.

## 4 - Training
Setup hardware e software; Iperparametri e range vari. Spiegazione della logica che ho adottato per raccogliere i vari risultati e fare i test.

## 5 - Results
Sequenza di plot con relativa spiegazione.

## 6 - Conclusion
Recap generale dell'esperienza, menzionando i risultati ottenuti da ciascun metodo.
