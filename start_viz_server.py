import socketio

# create a Socket.IO server
sio = socketio.AsyncServer()

@sio.event
def on_new_frame(_, data):
    sio.emit('on_new_frame', data=data)