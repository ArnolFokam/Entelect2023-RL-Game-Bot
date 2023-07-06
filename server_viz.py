from collections import defaultdict, deque


class ServerViz:
    def __init__(self, port: int) -> None:
        self.rooms = defaultdict(lambda : deque(maxlen=1000))
        
    def publish(self, room, frame):
        self.rooms[room].append(frame)