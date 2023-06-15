import numpy as np
from torch import Tensor


class BaseAgent:
    """A wrapper module for our learning agent."""
    
    def __init__(self, cfg, env, *args, **kwargs) -> None:
        self.cfg = cfg
        self.num_actions = env.action_space.n
        
        # initialize the env and info about the env
        self.env = env
        state = self.env.reset()
        assert isinstance(state, (np.ndarray, Tensor)), "state space must a numpy array or a tensor"
        self.observation_shape = state.shape
        
    def learn(self):
        raise NotImplementedError
        
    def save(self):
        raise NotImplementedError
        
    def load(self):
        raise NotImplementedError