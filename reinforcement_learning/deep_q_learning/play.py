#!/usr/bin/env python3
"""
Displays a game of Atari's Breakout played by the agent trained by train.py
"""


import gym
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
train = __import__('train')


if __name__ == '__main__':
    env = gym.make(train.ENV_NAME)
    nb_actions = env.action_space.n
    model = train.build_model(nb_actions)
    memory = SequentialMemory(limit=1, window_length=train.WINDOW_LENGTH)
    # always pick the action with the highest Q-value
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   test_policy=policy, memory=memory,
                   processor=train.AtariProcessor())
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=1, visualize=True)
