# CartPole-v0

## Overview

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0) is a Classic control environment. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.

- `Observation`: A array of 4 parameters denoting Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip.
- `Action`: The system is controlled by applying a force of +1 or -1 to the cart.
- `Reward`: A reward of +1 is provided for every timestep that the pole remains upright.
- `Terminal`: The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
- `Solved`: CartPole-v0 defines solved as getting average reward of 195.0 over 100 consecutive trials.

## Algorithms

### Random Search

A list of 4 random values in [-1, 1] are generated for each observation. The weighted sum of parameters and random values are considered for taking a new action. Since the action space is binary, 0 is selected as a threshold of taking actions. 

This process is carried until correct set of weights are selected and they are carried henceforth.

### Q-Learning

Q-Learning in a tabular fashion is implemented. State space is discretized to account for infinitely many states. Each state feature is divided into several (~10) bins, ranging from `feature_min` to `feature_max`. 

The policy followed by the agent is an ε-greedy policy, where at each step, the action seleced is from the `Q table` ε-greedily. Further the updated of Q-values are carried out at **each step** according to

```
td_target = reward + gamma * max(Q[next_state])
td_error = td_target - Q[state][action]
Q[state][action] += alpha * td_error
```

where, `gamma` is discount factor and `alpha` is the step size.
