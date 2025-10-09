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

        raw = env
        while hasattr(raw, 'env'):
            raw = raw.env
        self.raw_env = raw

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

        self.prev_state = None

    def _state_key(self, obs):
        return tuple(obs)
    
    def get_action(self, obs, exploit_only=False):
        state = self._state_key(obs)
        valid = self.raw_env.valid_actions()  # only consider valid actions
        if not exploit_only and np.random.rand() < self.epsilon:
            return np.random.choice(valid)
        
        # Exploitation
        q_vals = self.q_values[state]
        q_vals_valid = {a: q_vals[a] for a in valid}

        # Find all actions with the maximum Q-value
        best_q = max(q_vals_valid.values())
        best_actions = [a for a, q in q_vals_valid.items() if q == best_q]

        # Randomly choose among the best actions (tie-breaking)
        action = np.random.choice(best_actions)
        return action

    def update(self, obs, action, reward, terminated, next_obs):
        state = self._state_key(obs)
        next_state = self._state_key(next_obs)

        # if self.prev_state is not None and next_state == self.prev_state:
        #     # Penalize for staying in the same state
        #     reward -= 0.7  # Adjust penalty as needed

        future_q = (not terminated) * np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q
        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error
        self.training_error.append(td_error)

        self.prev_state = state if not terminated else None

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

    print("Training complete! in agent file not learning file")
