
import os
import sys
import argparse
import numpy as np
from omegaconf import open_dict, OmegaConf

import torch

from ecbot.agents import agents
from ecbot.envs import environments

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)


def play(dir):
    print(f"Playing online with agent at {dir}")
    
    cfg = OmegaConf.load(os.path.join(dir, "config.yaml"))
    
    # no rendering when playing online
    with open_dict(cfg):
        cfg.render_mode = None
        cfg.game_server_port = 5000
    
    # Use a separate environement for evaluation
    env = environments[cfg.env_name](cfg, run="online")
    obs, done = env.online_reset()
    
    # load the model
    agent = agents[cfg.agent_name].load_trained_agent(cfg, dir, env)
    
    # game receives the input, before training
    try:
        while not done:
            obs = torch.tensor(np.array([obs]), dtype=torch.float32)
            action = agent.act(obs)
            obs, _, done, _ = env.online_step(action)
            # TODO: print cumulative rewards for debugging
    except KeyboardInterrupt:
        env.game_client.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    args = parser.parse_args()
    play(dir=args.model_path)
