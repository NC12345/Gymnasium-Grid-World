import gymnasium as gym
import gymnasium_env  # this registers GridWorld-v0
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt

from agents.qlearning_agent import GridWorldAgent  # import your agent

# 1 Create and wrap the environment
env = gym.make("gymnasium_env/GridWorld-v0", size=10)
env = FlattenObservation(env)  # flatten dict obs into 1D array

n_episodes = 50000

# 2 Create the agent
agent = GridWorldAgent(
    env=env,
    learning_rate=0.4,
    initial_epsilon=1.0,
    epsilon_decay=1.0 / (n_episodes / 2),
    final_epsilon=0.05,
    discount_factor=0.95,
)

# 3 Training loop
episode_rewards = []  # ← track total reward per episode

for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)
        obs = next_obs
        done = terminated or truncated
        total_reward += reward

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}/{n_episodes} — epsilon={agent.epsilon:.3f}")


# Check how many unique states were visited
visited_states = len(agent.q_values)
print("Unique states visited:", visited_states)

np.save("q_table.npy", dict(agent.q_values))
print("Q-table saved!")

env.close()
print("Training complete")

# 4 Plot learning curve
window = 100
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.3, label='Episode reward')
plt.plot(range(window-1, n_episodes), moving_avg, label=f'{window}-episode moving avg', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Training Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()