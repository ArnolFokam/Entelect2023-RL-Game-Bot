<h1 align="center">Entelect2023 RL Game Bot</h1>

<h2 align="center">
  <p>A Reinforcement Learning Bot to Play a Multiplayer Platformer Game</p>
</h2>

<div align="center">
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/python-3.8-blue" alt="Python version 3.8"/>
  </a>
</div>

<div align="center">
  <h3>
    <a href="https://challenge.entelect.co.za/home">Enlect Challenge</a> |
    <a href="#overview">Overview</a> |
    <a href="#quickstart">Quickstart</a> |
    <a href="#usage">Usage</a> |
    <a href="#contributing">Contributing</a> |
    <a href="#author">Author's Info</a>
  </h3>
</div>

<h2 name="overview" id="overview">Overview </h2>

This project showcases a bot trained using reinforcement learning algorithms such as DQN and PPO, developed from scratch to play a platformer game. The game is hosted on a server implemented by Entelect, which can be accessed at this [link](https://github.com/EntelectChallenge/2023-Cy-Fi). However, an OpenAI Gym wrapper environment is utilized to communicate with the server for training the bot.

<h2 name="quickstart" id="quickstart">Quickstart </h2>

### System Requirements

* Docker
* .NET = `6.0`
* Python >= `3.8`

The following steps assume the repository is already cloned and you are on a terminal with a working Python environment.

### Install Project Requirements

Here's the corrected version:

* Install `dotnet` by following these [instructions](https://learn.microsoft.com/en-us/dotnet/core/install/linux-scripted-manual#scripted-install).
* Make sure the `dotnet` command can be called from the terminal (see [here](https://learn.microsoft.com/en-us/dotnet/core/install/linux-scripted-manual#example) for help).
* Install PyTorch by following the steps outlined [here](https://pytorch.org/get-started/locally/).
* Install project requirements from the specified file.

```bash
pip install -r requirements.txt
```

<h2 name="usage" id="usage">Usage 🪛</h2>

### Train an Agent on the Game Server

* Start the server with the following command:

```bash
cd game/2023-CyFi
dotnet restore
dotnet publish --configuration Release --output ./publish
cd publish && dotnet CyFi.dll <PORT_YOU_WANT_TO_USE_SERVER_ON>
```

* The console will print something similar to what you see below:

```bash
2024-04-22T02:51:09.8400811+01:00 Information  - Starting System.Private.CoreLib
SignalR Confighttp://127.0.0.1:5000/runnerhub
COLLECTIBLE AMOUNT: 270
HAZARD AMOUNT: 153
Start position {X=291,Y=123} 
COLLECTIBLE AMOUNT: 540
HAZARD AMOUNT: 136
Start position {X=456,Y=124} 
COLLECTIBLE AMOUNT: 351
HAZARD AMOUNT: 335
Start position {X=156,Y=53} 
COLLECTIBLE AMOUNT: 459
HAZARD AMOUNT: 219
Start position {X=143,Y=48} 
2024-04-22T02:51:10.5407311+01:00 Information Runner.Services.CloudIntegrationService - Cloud Callback Initiated, Status: initializing, Callback player Count: 0
2024-04-22T02:51:10.5644055+01:00 Warning Runner.Services.CloudIntegrationService - Failed to make cloud callback with error: An invalid request URI was provided. Either the request URI must be an absolute URI or BaseAddress must be set.
2024-04-22T02:51:10.7034580+01:00 Information Microsoft.Hosting.Lifetime - Now listening on: "http://127.0.0.1:5000"
2024-04-22T02:51:10.7111390+01:00 Information Microsoft.Hosting.Lifetime - Application started. Press Ctrl+C to shut down.
2024-04-22T02:51:10.7120336+01:00 Information Microsoft.Hosting.Lifetime - Hosting environment: "Production"
2024-04-22T02:51:10.7123491+01:00 Information Microsoft.Hosting.Lifetime - Content root path: "/home/arnol/projects/rlbot/game/2023-CyFi/publish"
```

* While the server is running, train the agent with the following command

```bash
python train.py --config-path=exps --config-name=ppo $(python new_hydra_dir_params.py) game_server_port=<PORT_YOU_WANT_TO_USE_SERVER_ON>
```

* The output of the training command will be similar to what you see below

```bash
Starting connection...
Connection started
socket server is not initialized.
Starting 1th episode
Registering bot
Bot registered with ID:  74e8ac32-004b-11ef-9364-f77699ea19da
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: -0.5, mean: -0.5
reward: 7.5, mean: 0.3888888888888889
```

### Visualize an agent training

* The following steps assumes that the server is running on the port 5000
* First run the visualization server with the following command

```bash
python start_viz_server.py game_server_port=<VIZ_SERVER_PORT>
```

<h2 name="contributing" id="contributing">Contributing 🤝</h2>

I do not accept contributions for now, but feel free to raise any issues you spot with the source code.

<h2 name="author" id="author">Author's Info 👨‍🎨</h2>

* Website: https://arnolfokam.github.io/
* Twitter: [@ArnolFokam](https://twitter.com/ArnolFokam)
* LinkedIn: [@arnolfokam](https://linkedin.com/in/arnolfokam)

**Note:** Do all your dev work on the train branch.

# Requires
- Python 3.8.16
- Docker
- Conda (Anaconda or Miniconda)

# How to run

## Environment

```bash
cd game/2023-CyFi
dotnet restore
dotnet publish --configuration Release --output ./publish
cd publish && dotnet CyFi.dll [port you want the server to run on]
```

## Agent
- Create python environment with `conda create -n ecbot python=3.8.16`
- Activate the environment with `conda activate ecbot`
- Install python things with `pip install -r requirements.txt`
- While env is running (on another window), run the following command:

```bash
python train.py --config-path=exps --config-name=ppo $(python new_hydra_dir_params.py) game_server_port=[port on which the server is running]
```

Note: if pygame crashes on some iris, swarst issues run the following command:

```bash
conda install -c conda-forge libstdcxx-ng
``` 
