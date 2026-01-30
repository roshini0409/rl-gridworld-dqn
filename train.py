import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from envs.gridworld_env import GridWorldEnv
from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer

# ------------------- Reproducibility -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def train():
    # ------------------- Environment -------------------
    env = GridWorldEnv(grid_size=5)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # ------------------- Hyperparameters -------------------
    episodes = 300
    batch_size = 64
    gamma = 0.99
    lr = 0.001

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    target_update_freq = 10

    # ------------------- Device -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------- Networks -------------------
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ------------------- Replay Buffer -------------------
    memory = ReplayBuffer(10000)

    rewards_per_episode = []
    best_reward = -float("inf")

    # ------------------- Training Loop -------------------
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(state)).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)

            memory.push(
                state.cpu().numpy(),
                action,
                reward,
                next_state.cpu().numpy(),
                done
            )

            state = next_state
            total_reward += reward

            # Learn when enough samples are available
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                loss = criterion(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ------------------- End of Episode -------------------
        rewards_per_episode.append(total_reward)

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), "best_model.pth")

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    # ------------------- Save Final Model -------------------
    torch.save(policy_net.state_dict(), "policy_model.pth")

    # ------------------- Plot Results -------------------
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_per_episode, label="Episode Reward")
    plt.plot(
        moving_average(rewards_per_episode),
        label="Moving Average (10)",
        linewidth=2
    )
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.legend()
    plt.savefig("plots/training_rewards.png")
    plt.show()


if __name__ == "__main__":
    train()
