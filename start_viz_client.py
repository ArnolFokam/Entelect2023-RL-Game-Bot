import socketio

sio = socketio.AsyncClient()

@sio.event
def on_new_frame(data):
    print(data)
    
if __name__ == "__main__":
    sio.connect('http://localhost:6000')