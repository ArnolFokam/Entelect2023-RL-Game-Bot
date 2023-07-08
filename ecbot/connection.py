import os
import uuid
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict,  List, Optional


from signalrcore.helpers import logging as LogLevel
from signalrcore.hub_connection_builder import HubConnectionBuilder


class Constants:
    CURRENT_LEVEL="currentLevel"
    CONNECTION_ID="connectionId"
    COLLECTED="collected"
    ELAPSED_TIME="elapsedTime"
    HERO_WINDOW="heroWindow"
    POSITION_X="x"
    POSITION_Y="y"

@dataclass
class State:
    connected: bool = False
    
    # id of the bot in the game
    bot_id: Optional[List[str]]= None
    
    # state of the bot
    bot_state: List[Dict[str, Any]] = field(default_factory=list) 
    
    # game is completed
    game_completed: bool = False

class CiFyClient:
    
    
    def __init__(self, port: int = 5000, max_frames: Optional[int] = 10) -> None:
        # Configuration
        runner_ip = os.getenv("RUNNER_IPV4") or "localhost"
        runner_ip = runner_ip if runner_ip.startswith("http://") else f"http://{runner_ip}"

        # Build SignalR connection to Runner Hub
        self.connection = (
            HubConnectionBuilder()
            .with_url(f"{runner_ip}:{port}/runnerhub")
            .configure_logging(LogLevel.ERROR)
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
            if len(self.state.bot_state) >= max_frames:
                self.state.bot_state.pop(0)
            self.state.bot_state.append(params[0])
        
        # When the GameCompleted command is sent from the runner.
        def on_game_completed():
            self.state.game_completed = True
            print("Game completed")
                

        self.connection.on_open(on_open)
        self.connection.on_close(on_close)
        self.connection.on("Disconnect", on_disconnect)
        self.connection.on("Registered", on_registered)
        self.connection.on("ReceiveBotState", on_receive_bot_state)
        self.connection.on("GameCompleted", on_game_completed)
        
        # connect to the game server
        self.connect()
        
    def send_player_command(self, action: int):
        self.connection.send("SendPlayerCommand", [{
            "Action" : action,                                   
            "BotId" : self.state.bot_id,
        }])
        time.sleep(0.1)
        
    def register_new_player(self):
        # register new bot
        print("Registering bot")
        bot_nickname = os.getenv("BOT_NICKNAME") or f"AAIIGBot-{uuid.uuid1()}"
        self.connection.send("Register", [bot_nickname])
        time.sleep(0.1)
        
    def connect(self):
        # initiate connection with the game server
        print("Starting connection...")
        self.connection.start()
        time.sleep(0.1)
        
    def disconnect(self):
        print("Disconnecting bot...")
        self.connection.stop()
        time.sleep(0.1)
        
    def new_game(self):
        self.connection.send("RestartGame", [])
        time.sleep(0.1)
        self.register_new_player()