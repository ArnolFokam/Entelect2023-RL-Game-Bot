import hydra
import wandb
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import omegaconf

from ecbot.envs import environments
from ecbot.agents import agents
from ecbot.helpers import generate_random_string, get_dir

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.project)
    wandb.define_metric("episode")
    wandb.run.define_metric("train-episode-reward", step_metric="episode", goal="maximize")
    wandb.run.define_metric("eval-mean-reward", step_metric="episode", goal="maximize")
    wandb.run.define_metric("eval-min-behaviour", step_metric="episode")
    wandb.run.define_metric("eval-min-behaviour", step_metric="episode")

    env = environments[cfg.env_name](cfg=cfg)
    agent = agents[cfg.agent_name](env=env, cfg=cfg)
    
    # train agent
    agent.learn()
    
    # save agent
    saved_agent_dir = get_dir(f"{HydraConfig.get().runtime.output_dir}/{cfg.env_name}_{cfg.agent_name}_{generate_random_string(5)}")
    agent.save(saved_agent_dir)

if __name__ == "__main__":
    main()