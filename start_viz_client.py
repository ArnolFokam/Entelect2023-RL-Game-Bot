import sys
import socketio
import pygame
import numpy as np

from ecbot.envs.cyfi import CyFi
from ecbot.helpers import bytes_to_array

sio = socketio.Client()
frames = []

@sio.event
def on_new_frame(data):
    run = data["run"]
    frame_buffer = data["frame"]
    frames.append((run, bytes_to_array(frame_buffer)))
    
if __name__ == "__main__":
    
    assert len(sys.argv) == 3

    # initialize pygame on environment manually
    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode((
        CyFi.window_width * CyFi.block_size,
        CyFi.window_height * CyFi.block_size,
    ))
    clock = pygame.time.Clock()

    
    sio.connect(f'http://localhost:{sys.argv[1]}')
    viz_run = sys.argv[2]
    
    # Main viz loop
    running = True
    while running:
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        if len(frames) > 0:
            
            # get the game surface
            run, frame = frames.pop(0)
            
            if run == viz_run:
                frame = np.rot90(frame, k=1) 
                frame = np.flip(frame, axis=0)
                surface = pygame_surface = pygame.surfarray.make_surface(frame)

                # Rendering
                window.blit(surface, (0, 0))
                pygame.display.update()

    # Clean up
    pygame.display.quit()
    pygame.quit()
    sio.disconnect()