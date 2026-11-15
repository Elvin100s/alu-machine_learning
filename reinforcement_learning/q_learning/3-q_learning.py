#!/usr/bin/env python3
"""
Defines function that performs Q-learning
"""


import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    parameters:
        env: the FrozenLakeEnv instance
        Q [numpy.ndarray]:
            contains the Q-table
        episodes [int]:
            the total number of episodes to train over
        max_steps [int]:
            the maximum number of steps per episode
        alpha [float]:
            the learning rate
        gamma [float]:
            the discount rate
        epsilon [float]:
            the initial threshold for epsilon greedy
        min_epsilon [float]:
            the minimum value that epsilon should decay to
        epsilon_decay [float]:
            the decay rate for updating epsilon between episodes

    returns:
        Q, total_rewards:
            Q [numpy.ndarray]: the updated Q-table
            total_rewards [list]: contains the rewards per episode
    """
    initial_epsilon = epsilon
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            # falling in a hole is penalized
            if env.unwrapped.desc.flat[next_state] == b'H':
                reward = -1
            # Bellman equation
            Q[state, action] += alpha * (reward +
                                         gamma * np.max(Q[next_state]) -
                                         Q[state, action])
            state = next_state
            episode_rewards += reward
            if done:
                break
        # decay epsilon exponentially towards min_epsilon
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
        total_rewards.append(episode_rewards)
    return Q, total_rewards
