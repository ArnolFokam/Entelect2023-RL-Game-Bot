import hydra
from omegaconf import DictConfig

from hydra.core.hydra_config import HydraConfig
from stable_baselines3.common.evaluation import evaluate_policy

from ecbot.envs import environments
from ecbot.agents import agents
from ecbot.helpers import generate_random_string, get_dir

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    env = environments[cfg.env_name](cfg)
    agent = agents[cfg.agent_name](cfg, env)
    
    # train agent
    agent.learn(
        total_timesteps=cfg.training_steps, 
        progress_bar=True
    )
    
    # save agent
    output_dir = get_dir(HydraConfig.get().runtime.output_dir)
    saved_agent_dir = f"{output_dir}/{cfg.env_name}_{cfg.agent_name}_{generate_random_string(5)}"
    agent.save(saved_agent_dir)
    
    # Load the trained agent
    model = agents[cfg.agent_name].load(saved_agent_dir)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        model, 
        env,
        cfg.render,
        cfg.deterministic,
        cfg.n_eval_episodes,
    )

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()