import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium_env.wrappers import RelativePosition


env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human",size=5)
# env = FlattenObservation(env)
env = RelativePosition(env)
obs, info = env.reset()

for _ in range(30):
    action = env.action_space.sample()  # move randomly
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {_}: obs = {obs}, reward = {reward}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()