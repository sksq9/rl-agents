# MountainCar-v0

## Overview

[MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0) is a Classic control environment and probably one of the most cited RL environment. A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

## Algorithms

### Q-Learning

Q-Learning with function approximation. Stochastic Gradient Descent is used with Least Square Error to minimize difference between predicted `Q[state][action]` and Q-value in accordance with Q-Learning.

Input features are normalized to zero mean and unit variance and further passed through 4 different RBF kernels with varying variance. A seperate model is use for each action in practice. At **each step** the model is update to fit the predicted `td_target` according to:

```
td_target = reward + gamma * max(Q.predict(next_state))
Q.update(state, action, td_target) # SGD model fitting
```
