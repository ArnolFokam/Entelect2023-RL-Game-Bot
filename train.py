import os
import sys
import wandb
import hydra
import torch
import random
import omegaconf
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


from ecbot.agents import agents
from ecbot.envs import environments
from ecbot.helpers import generate_random_string, get_dir, get_new_run_dir_params, has_valid_hydra_dir_params

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    results_dir = get_dir(HydraConfig.get().runtime.output_dir)
    run_name = os.path.basename(results_dir)
    
    # ensure reprodcibility and speed up
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    wandb.init(
        project=cfg.project,
        name=run_name,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )
    
    # wandb stuffs
    wandb.define_metric("episode")
    wandb.run.define_metric("train-episode-reward", step_metric="episode", goal="maximize")
    wandb.run.define_metric("train-actor-loss", step_metric="episode", goal="minimize")
    wandb.run.define_metric("train-critic-loss", step_metric="episode", goal="minimize")
    wandb.run.define_metric("eval-mean-reward", step_metric="episode", goal="maximize")
    wandb.run.define_metric("eval-min-behaviour", step_metric="episode")
    wandb.run.define_metric("eval-min-behaviour", step_metric="episode")

    env = environments[cfg.env_name](cfg=cfg, run=run_name)
    agent = agents[cfg.agent_name](env=env, cfg=cfg)
    
    # train agent
    agent.learn()
    
    # save agent
    saved_agent_dir = get_dir(f"{HydraConfig.get().runtime.output_dir}/{cfg.env_name}_{cfg.agent_name}_{generate_random_string(5)}")
    agent.save(saved_agent_dir)

if __name__ == "__main__":
    if has_valid_hydra_dir_params(sys.argv):
        main()
    else:
        params = get_new_run_dir_params()
        for param, value in params.items():
            sys.argv.append(f"{param}={value}")
        main()