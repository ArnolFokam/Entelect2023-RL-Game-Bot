import sys
import socketio
from aiohttp import web

# create a Socket.IO server
sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

@sio.event
async def new_frame(_, data):
    await sio.emit('on_new_frame', data=data)
    
if __name__ == '__main__':
    web.run_app(app, port=sys.argv[1])