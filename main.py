import hydra
from omegaconf import DictConfig

from ecbot.envs import environments
from ecbot.agents import agents

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    env = environments[cfg.env_name](cfg=cfg)
    agent = agents[cfg.agent_name](env=env, cfg=cfg)
    
    # train agent
    agent.learn()
    
    # save agent
    # TODO: write the code to save the learned agent
    # output_dir = get_dir(HydraConfig.get().runtime.output_dir)
    # saved_agent_dir = f"{output_dir}/{cfg.env_name}_{cfg.agent_name}_{generate_random_string(5)}"
    # agent.save(saved_agent_dir)
    
    # Load the trained agent
    # TODO: write the code to load the learned agent
    # model = agents[cfg.agent_name].load(saved_agent_dir)

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