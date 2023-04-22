import numpy as np

from gym import Env
from gym import spaces
import pygame

from ecbot.connection import CiFyClient, Constants

class SinglePlayerSingleAgentEnv(Env):
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
    
    def __init__(self, render_mode=None):
        self.game_server = CiFyClient()
        
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
        self.action_space = spaces.Discrete(12, start=1)
        
        # Our CiFy agent can only observe a window 
        # of height of 33 and a weight of 20 arount it
        # Note: the agent is at the center of the window
        self.observation_space = {
            "window": spaces.Box(low=0, high=6, shape=(33, 20), dtype=int)
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # window and clock rate for 
        # the defined rendering modes
        self.window = None
        self.clock = None
    
    def step(self, action: int):
        self.game_server.send_player_command(action)
        
        self._wait_for_game_state()
        observation, info, done = self._return_env_state()
        
        # TODO: Add reward function
        reward = 0
        
        return observation, reward, done, info
    
    def render(self, observation):
        frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        if self.render_mode == "human":
            
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
            
            for x in range(self.window_width):
                for y in range(self.window_height):
                    frame[y, x] = self.cellToColor[observation["window"][y][x]]
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
            
        elif self.render_mode == "rgb_array":
            return frame
    
    def reset(self):
        # TODO: Handle case where game is already full
        # TODO: Add endpoint on server to restart the game
        self.game_server.reconnect()
        
        self._wait_for_game_state()
        observation, info, _ = self._return_env_state()
        
        if self.render_mode == "human":
            self.render(observation)
        
        return observation, info
    
    def close(self):
        self.game_server.disconnect()
        
        # detach pygame if it was initialised
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
    def _return_env_state(self):
        game_state = self.game_server.state.bot_state.pop(0)
        return self._get_observation(game_state), self._get_info(game_state), self.game_server.state.completed
    
    def _wait_for_game_state(self):
        while not len(self.game_server.state.bot_state) > 0:
            pass
        
    def _get_observation(self, game_state):
        return {
            "window": game_state[Constants.HERO_WINDOW]
        }
        
    def _get_info(self, game_state):
        return {
            "collected": game_state[Constants.COLLECTED],
            "position": (
                game_state[Constants.POSITION_X],
                game_state[Constants.POSITION_Y],
            ),
            "level": game_state[Constants.CURRENT_LEVEL],
            "elapsed_time": game_state[Constants.ELAPSED_TIME],
        }