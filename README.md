# Reinforcement Learning – GridWorld using DQN

## Overview
This project was developed and tested on Python 3.11 using a virtual environment.

This project implements a **custom GridWorld environment** and trains an agent using a **Deep Q-Network (DQN)** to learn an optimal navigation policy.  
The agent learns to reach a goal while avoiding obstacles using reinforcement learning principles.

The entire system is implemented **from scratch** using **Python, OpenAI Gym, and PyTorch**, without relying on pre-built environments.

---

## Environment Description
- Grid size: **5 × 5**
- Agent starts at the **top-left corner**
- Goal is located at the **bottom-right corner**
- Fixed obstacles are placed within the grid

### State Representation
- The environment state is represented as a **flattened grid**
- Each cell is numerically encoded as:
  - `0` → Empty cell  
  - `1` → Agent position  
  - `2` → Obstacle  
  - `3` → Goal  

State dimension = `grid_size × grid_size`

This encoding allows the neural network to distinguish between different environment elements.

---

### Action Space
The agent operates in a **discrete action space** with 4 possible actions:
- `0` → Move Up
- `1` → Move Down
- `2` → Move Left
- `3` → Move Right

---

## Reward Function
The reward function is designed using **reward shaping** to guide efficient learning:
- `+5` → Reaching the goal
- `-1` → Hitting an obstacle
- `-0.1` → Step penalty to encourage shorter paths

This reward structure promotes goal-directed and efficient navigation.

---

## DQN Architecture
The Deep Q-Network (DQN) consists of:
- A fully connected neural network
- Two hidden layers with ReLU activation
- An output layer producing Q-values for each action

### Techniques Used
- **Experience Replay** to break correlation between consecutive samples
- **Target Network** for training stability
- **Epsilon-greedy exploration** to balance exploration and exploitation

---

## Training Details
- Number of episodes: **300**
- Discount factor (γ): **0.99**
- Learning rate: **0.001**
- Optimizer: **Adam**
- Loss function: **Mean Squared Error (MSE)**

During training:
- The agent initially explores randomly
- The exploration rate (epsilon) decays gradually
- The target network is updated periodically to stabilize learning

---

## Results
The training reward curve demonstrates effective learning behavior. In the initial episodes, the agent receives low and highly variable rewards due to random exploration and frequent suboptimal actions. As training progresses and the exploration rate (epsilon) decays, the agent increasingly exploits learned policies, leading to a steady improvement in cumulative rewards. The moving average of episode rewards highlights a clear upward trend and eventual stabilization, indicating convergence to a reliable and efficient navigation strategy. Minor fluctuations persist due to residual exploration, which is expected in epsilon-greedy learning.


A **reward vs episode plot** is generated after training to visualize learning progress.

---

## How to Run

### 1. Activate the virtual environment
```bash
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the agent
```bash
python train.py
```
### 3.Evaluate the trained agent
```bash
python evaluate.py
```

