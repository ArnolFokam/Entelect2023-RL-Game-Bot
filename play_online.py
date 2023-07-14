
import sys
import hydra
import numpy as np
from omegaconf import DictConfig, open_dict

import torch

from ecbot.agents import models
from ecbot.envs import environments



@hydra.main(version_base=None)
def play(cfg: DictConfig) -> None:
    
    # no rendering when playing online
    with open_dict(cfg):
        cfg.render_mode = None
    
    # Use a separate environement for evaluation
    env = environments[cfg.env_name](cfg)
    obs. done = env.reset()
    
    # load the model
    agent = models[cfg.agent_name].load_trained_agent(cfg.model_dir)
    
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
    play()
