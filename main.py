from enum import Enum
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridEnv(gym.Env):
    def __init__(self, render_mode=None, size=5):
        self.window_size = 512
        self.size = size

        