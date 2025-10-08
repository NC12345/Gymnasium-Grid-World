import gymnasium
from gymnasium.wrappers import FlattenObservation
from agents.qlearning_agent import GridWorldAgent  # Your trained agent
import numpy as np
from collections import defaultdict

# Create environment in human-render mode
env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human", size=10)
env = FlattenObservation(env)  # flatten dict obs into 1D array

# Load Agent
agent = GridWorldAgent(env=env)

# Load Q-table
loaded_q = np.load("gridworld_q_table.npy", allow_pickle=True).item()
agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), loaded_q)
print("Q-table loaded!")

episodes_to_watch = 5

for episode in range(episodes_to_watch):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Choose the best action (no exploration)
        action = agent.get_action(obs, exploit_only=True)  # implement flag in agent if needed
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
