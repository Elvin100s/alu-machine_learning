#!/usr/bin/env python3
"""
Defines function that trains a policy with the Monte-Carlo policy
gradient algorithm (REINFORCE)
"""


import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training with Monte-Carlo policy gradient

    parameters:
        env: the initial environment
        nb_episodes [int]:
            the number of episodes used for training
        alpha [float]:
            the learning rate
        gamma [float]:
            the discount factor
        show_result [bool]:
            if True, render the environment every 1000 episodes

    returns:
        [list]: the score (sum of all rewards) of every episode
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []
    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        done = False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, done, _ = env.step(action)
            state = state[None, :]
            grads.append(grad)
            rewards.append(reward)
        # walk the episode backwards to build each step's discounted return
        G = 0
        for grad, reward in zip(grads[::-1], rewards[::-1]):
            G = reward + gamma * G
            weight += alpha * G * grad
        score = sum(rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score), end="\r",
              flush=False)
    return scores
