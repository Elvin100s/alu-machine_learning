# Policy Gradients

This directory contains a Monte-Carlo policy gradient (REINFORCE) agent that
learns to balance OpenAI Gym's `CartPole-v1`.

| File | Description |
| --- | --- |
| `policy_gradient.py` | `policy`: softmax policy from a state and a weight matrix; `policy_gradient`: samples an action and returns the gradient of its log-probability |
| `train.py` | `train`: runs full episodes, then updates the weights with each step's discounted return; `show_result=True` renders every 1000th episode |

## How it works

- The policy is `softmax(state · W)`, giving a probability for each action.
- After sampling an action, the gradient of `log π(action | state)` with
  respect to `W` is `stateᵀ · (one_hot(action) − π)`.
- At the end of each episode, each step's gradient is scaled by its
  discounted return `G_t = r_t + γ·G_{t+1}` and added to `W` with
  learning rate `α`. Actions that led to long balancing runs become more
  likely.
