import sys
import socketio

sio = socketio.AsyncClient()

@sio.event
async def on_new_frame(data):
    print(data)
    
if __name__ == "__main__":
    sio.connect(f'http://localhost:{sys.argv[1]}')
    sio.emit("new_frame", {"yo": "yo"})