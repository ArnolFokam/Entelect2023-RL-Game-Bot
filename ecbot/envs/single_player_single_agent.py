import numpy as np

import gym
from gym import spaces

from ecbot.connection import Constants
from ecbot.envs.cyfi import CyFi

class SinglePlayerSingleAgentEnv(CyFi):
    
    def __init__(self, cfg):

        super().__init__(cfg)
        
        # Our CiFy agent has 12 possible actions
        # UP - 1
        # DOWN- 2
        # LEFT - 3
        # RIGHT - 4
        # UPLEFT - 5
        # UPRIGHT - 6
        # DOWNLEFT - 7
        # DOWNRIGHT - 8
        # DIGDOWN - 9
        # DIGLEFT - 10
        # DIGRIGHT - 11
        # STEAL - 12 (WIP wrench)
        self.action_space = gym.spaces.Discrete(12)
        
        # Our CiFy agent only knows about its windowed view of the world
        # Note: the agent is at the center of the window
        self.observation_space = spaces.Box(low=0, high=6, shape=(34 * 20,), dtype=int)
        
        # reward from collectables
        self.last_collected = 0
        self.current_level = 0
        
    def _calculate_reward(self, done):
        return -0.01 if not done else 1.0
        
    def step(self, action: int):
        # command from the  game server are 1-indexed
        self.game_client.send_player_command(int(action + 1))
        self._wait_for_game_state()
        
        self.observation, self.info, done = self._return_env_state()
        reward = self._calculate_reward(done)
        return self.observation, reward, done, self.info
        
    def _get_observation(self, game_state):
        return np.array(game_state[Constants.HERO_WINDOW], dtype=np.uint8).flatten()
        
    def _get_info(self, game_state):
        return {
            "position": (
                game_state[Constants.POSITION_X],
                game_state[Constants.POSITION_Y],
            ),
            "window": np.rot90(game_state[Constants.HERO_WINDOW]),
            "elapsed_time": game_state[Constants.ELAPSED_TIME],
            "collected": game_state[Constants.COLLECTED],
            "current_level": game_state[Constants.CURRENT_LEVEL]
        }