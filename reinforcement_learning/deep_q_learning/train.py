#!/usr/bin/env python3
"""
Trains a DQN agent to play Atari's Breakout with keras-rl
and saves the final policy network to policy.h5
"""


import gym
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


ENV_NAME = 'BreakoutDeterministic-v4'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """
    Preprocesses Breakout frames and rewards for the DQN agent
    """

    def process_observation(self, observation):
        """
        Converts an RGB frame to an 84x84 grayscale image

        parameters:
            observation [numpy.ndarray of shape (210, 160, 3)]:
                the raw frame returned by the environment

        returns:
            [numpy.ndarray of shape (84, 84)]: the processed frame as uint8
        """
        img = Image.fromarray(observation).resize(INPUT_SHAPE).convert('L')
        return np.array(img).astype('uint8')

    def process_state_batch(self, batch):
        """
        Scales a batch of stacked frames to [0, 1]

        parameters:
            batch [numpy.ndarray]: the batch of uint8 states

        returns:
            [numpy.ndarray]: the batch as float32 in [0, 1]
        """
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        """
        Clips the reward to [-1, 1] so every brick is worth the same

        parameters:
            reward [float]: the raw reward from the environment

        returns:
            [float]: the clipped reward
        """
        return np.clip(reward, -1., 1.)


def build_model(nb_actions):
    """
    Builds the convolutional Q-network from 'Playing Atari with Deep
    Reinforcement Learning'

    parameters:
        nb_actions [int]: the number of possible actions

    returns:
        the keras model mapping 4 stacked frames to one Q-value per action
    """
    model = Sequential()
    # keras-rl stacks frames first: (window, h, w) -> (h, w, window)
    model.add(Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n
    model = build_model(nb_actions)
    # replay memory of past transitions, sampled to decorrelate updates
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    # explore a lot at first, then anneal epsilon from 1 to 0.1
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   memory=memory, processor=AtariProcessor(),
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000, train_interval=4,
                   delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    # checkpoint regularly so a long run can be stopped without losing it
    checkpoint = ModelIntervalCheckpoint('policy.h5', interval=100000)
    dqn.fit(env, nb_steps=1750000, callbacks=[checkpoint],
            log_interval=10000, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
