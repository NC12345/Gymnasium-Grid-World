import gymnasium
import gymnasium_env  # <- triggers registration

env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human",size=5)
obs, info = env.reset()

for _ in range(30):
    action = env.action_space.sample()  # move randomly
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()