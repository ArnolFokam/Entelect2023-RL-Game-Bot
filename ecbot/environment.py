import time

from gym import Env
from gym.spaces import Discrete

from ecbot.connection import CiFyServer

class CyFiEnv(Env):
    def __init__(self):
        self.game_server = CiFyServer()
        
        # Our CiFy agent has 12 possible actions
        self.action_space = Discrete(12, start=1)
        
        # Out CiFy agent can only observe a window 
        # of width of 33 and a height of 20 arount it
        self.observation_space = Discrete(33 * 20)
    
    def step(self, action: int):
        reward = self.game_server.send_player_command(action)
        obs = self.game_server.state.bot_window
        done = False
        return obs, reward, done, {}
    
    def render(self):
        pass
    
    def reset(self):
        # TODO: Handle case where game is already full
        
        self.game_server.reconnect()
        
        # Wait for the bot window to be initialised
        while not self.game_server.state.bot_window:
            pass
        
        return self.game_server.state.bot_window