import gymnasium
from gymnasium.wrappers import FlattenObservation
from agents.qlearning_agent import GridWorldAgent
import numpy as np
from collections import defaultdict

# Create environment WITHOUT rendering for faster evaluation
env = gymnasium.make("gymnasium_env/GridWorld-v0", size=10)
env = FlattenObservation(env)

# Load Agent
agent = GridWorldAgent(env=env)
loaded_q = np.load("q_table_lr=0.4.npy", allow_pickle=True).item()
agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), loaded_q)
print("Q-table loaded!")

# Evaluation parameters
episodes_to_test = 10000
successes = 0
truncations = 0
optimal_episodes = 0
total_rewards = []

for episode in range(episodes_to_test):
    obs, info = env.reset()
    done = False
    total_reward = 0

    raw_env = env.unwrapped  # Access the raw environment to get agent and target locations
    optimal_steps = abs(raw_env._agent_location[0] - raw_env._target_location[0]) + abs(raw_env._agent_location[1] - raw_env._target_location[1])
    steps = 0

    while not done:
        action = agent.get_action(obs, exploit_only=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    total_rewards.append(total_reward)

    if terminated:
        successes += 1

        if steps == optimal_steps:
            optimal_episodes += 1
    elif truncated:
        truncations += 1

# Print evaluation results
print(f"\nEvaluation over {episodes_to_test} episodes:")
print(f"  ✅ Successful episodes: {successes}")
print(f"  ⏹ Truncated episodes:  {truncations}")
print(f"  📈 Success rate:       {successes / episodes_to_test:.2%}")
print(f"  🏅 Average reward:     {np.mean(total_rewards):.3f}")
print(f"  🎯 Optimal path rate:   {optimal_episodes} ({optimal_episodes / episodes_to_test:.2%})")