#!/usr/bin/env python3
"""
Defines function that loads the pre-made FrozenLakeEnv environment
from OpenAI's gym
"""


import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from OpenAI's gym

    parameters:
        desc [list of lists or None]:
            contains a custom description of the map to load
        map_name [str or None]:
            contains the name of the pre-made map to load
        is_slippery [bool]:
            determines if the ice is slippery

    if both desc and map_name are None, a randomly generated 8x8 map is loaded

    returns:
        the environment
    """
    env = gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
