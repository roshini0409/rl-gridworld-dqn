import torch
from envs.gridworld_env import GridWorldEnv
from agent.dqn import DQN


def evaluate():
    env = GridWorldEnv(grid_size=5)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load("policy_model.pth", map_location=device))
    model.eval()

    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)

    done = False
    total_reward = 0

    print("Evaluating trained agent...\n")

    while not done:
        with torch.no_grad():
            action = torch.argmax(model(state)).item()

        next_state, reward, done, _, _ = env.step(action)
        env.render()

        state = torch.FloatTensor(next_state).to(device)
        total_reward += reward

    print("\nTotal Reward:", total_reward)


if __name__ == "__main__":
    evaluate()
