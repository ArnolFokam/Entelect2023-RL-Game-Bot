import sys
import hydra
from omegaconf import DictConfig

import pygame

from ecbot.envs import environments

def check_keyboard_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
        
            if event.key == pygame.K_KP8:
                return 0 # UP
            elif event.key == pygame.K_KP2:
                return 1 # DOWN
            elif event.key == pygame.K_KP4:
                return 2 # LEFT
            elif event.key == pygame.K_KP6:
                return 3 # RIGHT
            elif event.key == pygame.K_KP7:
                return 4 # UPLEFT
            elif event.key == pygame.K_KP9:
                return 5 # UPRIGHT
            elif event.key == pygame.K_KP1:
                return 6 # DOWNLEFT
            elif event.key == pygame.K_KP3:
                return 7 # DOWNRIGHT
            elif event.key == pygame.K_s:
                return 8 # DIGDOWN
            elif event.key == pygame.K_a:
                return 9 # DIGLEFT
            elif event.key == pygame.K_d:
                return 10 # DIGRIGHT
            elif event.key == pygame.K_SPACE:
                return 11 # STEAL
            elif event.key == pygame.K_f:
                return 12 # RADAR
    return None

@hydra.main(version_base=None)
def play(cfg: DictConfig) -> None:
    # Use a separate environement for evaluation
    env = environments[cfg.env_name](cfg, run="play")
    env.reset()
    done = False
    
    # initialize pygame on environment manually
    pygame.init()
    pygame.display.init()
    env.window = pygame.display.set_mode((
        env.window_width * env.block_size,
        env.window_height * env.block_size,
    ))
    env.clock = pygame.time.Clock()

    # game receives the input, before training
    while not done:
        supported_input = check_keyboard_input()

        if supported_input is not None:
            # action = env.action_space.sample()
            _, _, done, _, _ = env.step(supported_input)
        
        env.render()
        
    print("Game Completed!")


if __name__ == "__main__":
    play()
