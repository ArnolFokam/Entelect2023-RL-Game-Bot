import sys
import socketio

import numpy as np

sio = socketio.Client()

@sio.event
def on_new_frame(data):
    frame_buffer = data["frame"]
    frame = np.frombuffer(frame_buffer)
    print(frame)
    
if __name__ == "__main__":
    sio.connect(f'http://localhost:{sys.argv[1]}')