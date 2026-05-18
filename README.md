# 🧠 Reinforcement Learning in GridWorld

### Q-Learning vs SARSA (Comparative Study)

---

## 📌 Project Overview

This project implements a **Reinforcement Learning (RL)** environment using a custom **GridWorld** and compares two fundamental algorithms:

* Q-Learning (Off-policy)
* SARSA (On-policy)

The agent learns to navigate from a start state to a goal while avoiding obstacles and minimizing penalties.

---

## 🎯 Objectives

* Build a custom GridWorld environment
* Implement Q-Learning and SARSA from scratch
* Compare learning performance and stability
* Visualize learned policies and value functions

---

## 📁 Project Structure

```
RL_Project/
│
├── env/
│   └── grid_world.py
│
├── algorithms/
│   ├── q_learning.py
│   └── sarsa.py
│
├── visualization/
│   ├── compare.py
│   ├── heatmap.py
│   ├── policy_visualization.py
│   ├── trajectory_animation.py
│   └── stability_plot.py
│
├── figures/   (generated plots)
└── README.md
```

---

## 🌍 Environment Description

* Grid Size: **3 × 4 (12 states)**
* Start State: `S (0,0)`
* Goal State: `G (2,3)`
* Obstacles: `X`

### Rewards:

* Goal: **+10**
* Obstacle: **-10**
* Step penalty: **-1**

---

## 🤖 Algorithms

### Q-Learning (Off-policy)

Updates using the maximum future Q-value:

```
Q(s,a) ← Q(s,a) + α [r + γ max(Q(s')) - Q(s,a)]
```

### SARSA (On-policy)

Updates using the actual next action:

```
Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
```

---

## ⚙️ Hyperparameters

* Learning rate (α): 0.1
* Discount factor (γ): 0.9
* Exploration rate (ε): 0.1
* Episodes: 500

---

## 📊 Visualizations

The project includes multiple analysis tools:

* 📈 Reward comparison (Q vs SARSA)
* 🔥 Q-value heatmap
* 🧭 Learned policy visualization
* 🎬 Trajectory simulation
* 📉 Stability analysis

All figures are saved in the `figures/` directory.

---

## 📷 Sample Outputs

After running visualization scripts, generated outputs include:

* Learning curves
* Heatmaps of state values
* Policy arrows grid
* Agent trajectory animation
* Stability plots across runs

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install numpy matplotlib seaborn
```

### 2. Run Q-Learning

```bash
python algorithms/q_learning.py
```

### 3. Run SARSA

```bash
python algorithms/sarsa.py
```

### 4. Run visualizations

```bash
python visualization/compare.py
python visualization/heatmap.py
python visualization/policy_visualization.py
python visualization/stability_plot.py
python visualization/trajectory_animation.py
```

---

## 📌 Key Insights

* Q-Learning converges faster to an optimal policy
* SARSA produces safer but more conservative behavior
* Reward shaping strongly impacts learning efficiency
* Visualization helps interpret policy behavior clearly

---

## 🧠 What I Learned

* Implementing RL algorithms from scratch
* Difference between on-policy and off-policy learning
* Importance of exploration vs exploitation
* How environment design affects convergence

---

## 📈 Future Improvements

* Add stochastic transitions
* Extend to larger grids
* Apply Deep Q-Network (DQN)
* Add obstacles dynamics
* Introduce multi-agent RL

---

## 👨‍💻 Author

Team Project - Reinforcement Learning Course

---

## 📜 License

For educational purposes only.
