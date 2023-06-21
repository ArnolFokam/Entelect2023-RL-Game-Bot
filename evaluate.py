import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from ecbot.envs import environments
from ecbot.agents import agents
from ecbot.helpers import generate_random_string, get_dir

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    env = environments[cfg.env_name](cfg=cfg)
    agent = agents[cfg.agent_name](env=env, cfg=cfg)
    
    # Load the trained agent
    model = agent.load(cfg.pretrained_dir)

    # Random Agent, before training
    # TODO: write code to evaluate the learned policy
    # mean_reward, std_reward = evaluate_policy(
    #     model, 
    #     env,
    #     cfg.render,
    #     cfg.deterministic,
    #     cfg.n_eval_episodes,
    # )
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()