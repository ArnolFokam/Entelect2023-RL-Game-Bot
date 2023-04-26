import numpy as np

import gym
import pygame
from gym import spaces

from ecbot.connection import CiFyClient, Constants

class SinglePlayerSingleAgentEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 60,
    }
    
    window_height = 33 # must be equal to the height of the window as defined in the hero window
    window_width = 20 # must be equal to the width of the window as defined in the hero window
    block_size = 16
    
    cellToColor = {
        0: (255, 255, 255), # air => white
        1: (150, 75, 0), # solid => brown
        2: (0, 0, 0), # collectible => black
        3: (255, 0, 0), # hazard => red
        4: (0, 0, 255), # platform => blue
        5: (0, 255, 0), # ladder => green
        6: (255, 165, 0), # opponent => orange
        7: (255, 255, 0), # hero bot => yellow
    }
    
    def __init__(self):
        self.game_client = CiFyClient()
        
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
        self.observation_space = spaces.Box(low=0, high=6, shape=(33 * 20,), dtype=int)
        
        # window and clock rate for 
        # the defined rendering modes
        self.window = None
        self.clock = None
        
        # reward from collectables
        self.last_collected = 0
        self.current_level = 0
    
    def step(self, action: int):
        # command from the  game server are 1-indexed
        self.game_client.send_player_command(int(action + 1))
        
        self._wait_for_game_state()
        self.observation, self.info, done = self._return_env_state()
        reward = self._calculate_reward(done)
        return self.observation, reward, done, self.info
    
    def _calculate_reward(self, done):
        return -0.01 if not done else 1
    

    def render(self, mode=None):
        assert mode is None or mode in self.metadata["render_modes"]
        frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        if mode == "human":
            
            # initialise pygame if not already initialised
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((
                    
                    self.window_width * self.block_size,
                    self.window_height * self.block_size,
                ))
                
            # initialise clock if not already initialised
            if self.clock is None:
                self.clock = pygame.time.Clock()
                
            canvas = pygame.Surface((
                self.window_width * self.block_size,
                self.window_height * self.block_size,
            ))
            canvas.fill((255, 255, 255))
            
        
        # draw the window and create numpy array    
        for x in range(self.window_width):
            for y in range(self.window_height):
                frame[y, x] = self.cellToColor[self.info["window"][y][x]]
                
                if mode == "human":
                    pygame.draw.rect(
                        canvas,
                        frame[y, x],
                        pygame.Rect(
                            x * self.block_size,
                            y * self.block_size, 
                            self.block_size, 
                            self.block_size
                        )
                    )
        if mode == "human": 
            # draw the bot     
            pygame.draw.rect(
                canvas, 
                self.cellToColor[7], 
                pygame.Rect(
                    (self.window_width // 2) * self.block_size, 
                    (self.window_height // 2) * self.block_size, 
                    # hero occupties 2x2 blocks
                    2 * self.block_size, 
                    2 * self.block_size
                )
            )
                
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            
        if mode == "rgb_array":
            return frame
    
    def reset(self):
        # There are two cases of reset:
        # collected obtained => no new game
        # level completed => new game
        
        if self.game_client.state.game_completed:
            self.game_client.new_game()
            
        # handle the case the when the game is full of players
        # you could for example throw an error or restart the game
        # what might be nice is to ask for user input about that
        # but only when when doing interactive training.
        
        self._wait_for_game_state()
        self.observation, self.info, _ = self._return_env_state()
        
        return self.observation
    
    def close(self):
        self.game_client.disconnect()
        
        # detach pygame if it was initialised
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
    def _return_env_state(self):
        # get more faithful game state from the previous
        game_state = self.game_client.state.bot_state.pop(-1)
        
        # get observation and info
        self.observation = self._get_observation(game_state)
        self.info = self._get_info(game_state)
        completed = self.game_client.state.game_completed
        
        if self.last_collected > self.info["collected"] or self.current_level < self.info["current_level"]:
            completed = True
        
        return self.observation, self.info, completed
        
    
    def _wait_for_game_state(self):
        while not len(self.game_client.state.bot_state) > 0:
            pass
        
    def _get_observation(self, game_state):
        return np.array(game_state[Constants.HERO_WINDOW], dtype=np.uint8).flatten()
        
    def _get_info(self, game_state):
        return {
            "position": (
                game_state[Constants.POSITION_X],
                game_state[Constants.POSITION_Y],
            ),
            "window": game_state[Constants.HERO_WINDOW],
            "elapsed_time": game_state[Constants.ELAPSED_TIME],
            "collected": game_state[Constants.COLLECTED],
            "current_level": game_state[Constants.CURRENT_LEVEL]
        }