# Q-learning

This directory contains a Q-learning agent that learns to play OpenAI Gym's
`FrozenLake` environment.

| File | Description |
| --- | --- |
| `0-load_env.py` | Loads a `FrozenLakeEnv` from a custom map, a pre-made map, or a random 8x8 map |
| `1-q_init.py` | Initializes the Q-table with zeros |
| `2-epsilon_greedy.py` | Chooses the next action with epsilon-greedy exploration |
| `3-q_learning.py` | Trains the Q-table with Q-learning, penalizing holes with -1 |
| `4-play.py` | Has the trained agent play an episode by always exploiting the Q-table |
