
import sys
import os
import numpy as np
from omegaconf import open_dict, OmegaConf

import torch

from ecbot.agents import models
from ecbot.envs import environments



def play(dir):
    
    cfg = OmegaConf.load(os.path.join(dir, "config.yaml"))
    
    # no rendering when playing online
    with open_dict(cfg):
        cfg.render_mode = None
        cfg.game_server_port = 5000
    
    # Use a separate environement for evaluation
    env = environments[cfg.env_name](cfg)
    obs. done = env.reset()
    
    # load the model
    agent = models[cfg.agent_name].load_trained_agent(cfg, dir, env)
    
    # game receives the input, before training
    try:
        while not done:
            obs = torch.tensor(np.array([obs]), dtype=torch.float32)
            action = agent.act(obs)
            obs, _, done, _ = env.online_step(action)
    except KeyboardInterrupt:
        env.game_client.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    play(dir=sys.argv[2])
