import sys

import pygame

from ecbot.envs.single_player_single_agent import SinglePlayerSingleAgentEnv

def check_keyboard_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
        
            if event.key == pygame.K_KP8:
                return 0
            elif event.key == pygame.K_KP2:
                return 1
            elif event.key == pygame.K_KP4:
                return 2
            elif event.key == pygame.K_KP6:
                return 3
            elif event.key == pygame.K_KP7:
                return 4
            elif event.key == pygame.K_KP9:
                return 5
            elif event.key == pygame.K_KP1:
                return 6
            elif event.key == pygame.K_KP3:
                return 7
            # DIGDOWN - 9
            # DIGLEFT - 10
            # DIGRIGHT - 11
            # STEAL - 12 
    return None


if __name__ == "__main__":
    # Use a separate environement for evaluation
    env = SinglePlayerSingleAgentEnv()
    obs = env.reset()
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
        print(supported_input)
        if supported_input is not None:
            # action = env.action_space.sample()
            obs, reward, done, info = env.step(supported_input)
        
        env.render(mode="human")

    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
