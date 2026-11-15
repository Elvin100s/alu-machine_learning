# Deep Q-learning

This directory contains a Deep Q-Network (DQN) agent that learns to play
Atari's Breakout with [keras-rl](https://github.com/keras-rl/keras-rl),
following *Playing Atari with Deep Reinforcement Learning* (Mnih et al.).

| File | Description |
| --- | --- |
| `train.py` | Trains a `DQNAgent` with `SequentialMemory` replay and an annealed `EpsGreedyQPolicy`, then saves the policy network to `policy.h5` |
| `play.py` | Loads `policy.h5` and displays a game played with a `GreedyQPolicy` |

## Requirements

python 3.5, numpy 1.15, gym 0.17.2 (with Atari), keras 2.2.5, keras-rl 0.4.2

## Usage

```
./train.py   # trains the agent and writes policy.h5
./play.py    # opens a window showing the trained agent play Breakout
```

## How it works

- Each frame is converted to an 84x84 grayscale image and the last 4 frames
  are stacked so the network can see the ball's motion.
- Rewards are clipped to [-1, 1].
- The policy network is a 3-layer CNN followed by a 512-unit dense layer
  with one Q-value output per action.
- Replay memory stores past transitions and samples random minibatches,
  which breaks the correlation between consecutive frames.
- A separate target network, updated every 10,000 steps, provides stable
  Q-learning targets.
