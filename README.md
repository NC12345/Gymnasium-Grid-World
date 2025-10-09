# Gymnasium GridWorld Q-Learning Agent
Training Performance:
![Alt text](res/Training%20Performance.png?raw=true "Training Performance")

Demo:
[Download Demo](res/Demo.mp4)

This project implements a custom **GridWorld environment** for reinforcement learning using **Gymnasium** and a **Q-learning agent**. The environment is a square grid where an agent navigates from a random starting location to a randomly placed target. The agent learns optimal paths through trial-and-error, receiving rewards for reaching the target and small penalties for each step to encourage efficiency.

## Features

### Custom Gymnasium environment
- Grid of configurable size
- Supports both **human-readable rendering** via PyGame and **fast evaluation mode**
- `step` and `reset` methods follow the Gymnasium API

### Q-learning agent
- Learns optimal policies over episodes
- Adjustable **learning rate (`lr`)** and **exploration rate (`epsilon`)**
- Supports **loading and saving Q-tables** for continued training or evaluation

### Reward shaping
- Positive reward for reaching the target
- Small step penalty to encourage shorter paths
- Optional backtracking penalty

### Evaluation metrics
- **Success rate**: Percentage of episodes where the agent reaches the target
- **Average reward**
- **Optimal path rate**: Percentage of episodes where the agent takes the shortest possible path

# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).


## Installation

To install your new environment, run the following commands:

```{shell}
cd gymnasium_env
pip install -e .
```

