import gymnasium as gym
import numpy as np
from collections import defaultdict
import gymnasium_env  # registers your GridWorld env

class GridWorldAgent:
    def __init__(
            self, 
            env, 
            learning_rate=0.1,
            initial_epsilon=1.0,
            epsilon_decay=1.0 / 5000,
            final_epsilon=0.05,
            discount_factor=0.95
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def _state_key(self, obs):
        return tuple(obs)

    def get_action(self, obs, exploit_only=False):
        state = self._state_key(obs)
        if exploit_only or np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    def update(self, obs, action, reward, terminated, next_obs):
        state = self._state_key(obs)
        next_state = self._state_key(next_obs)
        future_q = (not terminated) * np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q
        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


if __name__ == "__main__":
    env = gym.make("gymnasium_env/GridWorld-v0")
    agent = GridWorldAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=1.0 / 5000,
        final_epsilon=0.05,
    )

    n_episodes = 10000
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated

        agent.decay_epsilon()

    print("Training complete!")
