import numpy as np

import gym
import pygame

from ecbot.connection import CiFyClient

class CyFi(gym.Env):
    metadata = {
        "render_modes": ["human"], 
        "render_fps": 60,
    }
    
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
    
    window_height = 20 # must be equal to the height of the window as defined in the hero window
    window_width = 34 # must be equal to the width of the window as defined in the hero window
    block_size = 16 # block size per cell in the hero window
    
    def __init__(self, cfg) -> None:
        # save configurations
        self.cfg = cfg
        
        # initialise the game client
        self.game_client = CiFyClient()
        
        # window and clock rate for 
        # the defined rendering modes
        self.window = None
        self.clock = None
    
    def render(self, mode=None):
        """Rendering done only through PyGame"""
        assert mode is None or mode in self.metadata["render_modes"]
            
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
        
        # create a canvas to draw on
        canvas = pygame.Surface((
            self.window_width * self.block_size,
            self.window_height * self.block_size,
        ))
        canvas.fill((255, 255, 255))
            
        # draw the window and create numpy array    
        for x in range(self.window_width):
            for y in range(self.window_height):
                # draw the window
                if mode == "human":
                    pygame.draw.rect(
                        canvas,
                        self.cellToColor[self.info["window"][y][x]],
                        pygame.Rect(
                            x * self.block_size,
                            y * self.block_size, 
                            self.block_size, 
                            self.block_size
                        )
                    )
    
        # draw the bot     
        pygame.draw.rect(
            canvas, 
            self.cellToColor[7], 
            pygame.Rect(
                ((self.window_width // 2) - 1) * self.block_size, 
                ((self.window_height // 2)) * self.block_size,
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
    
    def _wait_for_game_state(self):
        while not len(self.game_client.state.bot_state) > 0:
            pass
    
    def _return_env_state(self):
        
        # get more faithful game state from the previous
        game_state = self.game_client.state.bot_state.pop(-1)
        
        # get observation and info
        self.observation = self._get_observation(game_state)
        self.info = self._get_info(game_state)
        completed = self.game_client.state.game_completed
        
        return self.observation, self.info, completed
    
    def _get_observation(self,):
        raise NotImplementedError
        
    def _get_info(self,):
        raise NotImplementedError
    
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