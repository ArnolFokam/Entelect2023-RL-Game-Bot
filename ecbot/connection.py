import os
import uuid
import time
import numpy as np
from dataclasses import dataclass
from typing import Any,  List, Optional


from signalrcore.helpers import logging as LogLevel
from signalrcore.hub_connection_builder import HubConnectionBuilder

@dataclass
class State:
    connected: bool = False
    bot_id: Optional[str] = None
    bot_window: Optional[np.ndarray] = None

class CiFyServer:
    def __init__(self) -> None:
        # Configuration
        runner_ip = os.getenv("RUNNER_IPV4") or "localhost"
        runner_ip = runner_ip if runner_ip.startswith("http://") else f"http://{runner_ip}"

        # Build SignalR connection to Runner Hub
        self.connection = (
            HubConnectionBuilder()
            .with_url(f"{runner_ip}:5000/runnerhub")
            .configure_logging(LogLevel.DEBUG)
            .with_automatic_reconnect(
                {
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                }
            )
            .build()
        )
        
        # Initialise state
        self.state = State()
        
        # When the connection starts
        def on_open():
            self.state.connected = True
            print("Connection started")

        # When the connection is closed
        def on_close(reason):
            self.state.connected = False
            print("Connection closed with reason: ", reason)

        # When the Disconnect command is sent from the runner.
        def on_disconnect(reason):
            self.state.connected = False
            print("Server sent disconnect command with reason: ", reason)

        # When the Registered command is sent from the runner.
        def on_registered(params: List[str]):
            self.state.bot_id = params[0]
            print("Bot registered with ID: ", self.state.bot_id)
            
        # When the ReceiveBotState commmand is sent from the runner.
        def on_receive_bot_state(params: List[Any]):
            self.state.bot_window = tuple([block for row in params[0]["heroWindow"] for block in row])
            
            # TODO: complete state of the connection
            # current and previous item collected
            # was an item stolen

        self.connection.on_open(on_open)
        self.connection.on_close(on_close)
        self.connection.on("Disconnect", on_disconnect)
        self.connection.on("Registered", on_registered)
        self.connection.on("ReceiveBotState", on_receive_bot_state)
        
    def send_player_command(self, action: int):
        self.connection.send("SendPlayerCommand", [{
            "Action" : action,                                   
            "BotId" : self.state.bot_id,
        }])
        time.sleep(0.2)
        
        # TODO: Implement reward function.
        # - Item collected: +1
        # - Item dropped: -1
        # - Item stolen: +1
        reward = 1
        return reward
        
        
    def connect(self):
        print("Starting connection...")
        self.connection.start()
        time.sleep(0.2)
        print("Registering bot")
        bot_nickname = os.getenv("BOT_NICKNAME") or f"AAIIGBot-{uuid.uuid1()}"
        self.connection.send("Register", [bot_nickname])
        time.sleep(0.2)
        
    def disconnect(self):
        print("Disconnecting bot...")
        self.connection.stop()
        time.sleep(0.2)
        
    def reconnect(self):
        self.disconnect()
        self.connect()