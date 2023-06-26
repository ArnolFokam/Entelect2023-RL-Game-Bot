import numpy as np


import torch
import hydra
from omegaconf import DictConfig

from ecbot.envs import environments
from ecbot.agents import agents

def evaluate(agent, env, render, n_eval_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    
    rewards = []
    
    for i_episode in range(n_eval_episodes):
        done = False
        episode_reward = 0.0
            
        print(f"Starting {i_episode + 1}th episode")
        
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action.item())
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward += reward
            
            if render:
                env.render(mode="human")
            
        rewards.append(episode_reward)
        print(f"Total Reward: {episode_reward}")
        
    return np.mean(rewards), np.std(rewards)
            

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    env = environments[cfg.env_name](cfg=cfg)
    agent = agents[cfg.agent_name](env=env, cfg=cfg)
    
    # Load the trained agent
    agent.load(cfg.pretrained_dir)
    
    mean_rewards, std_rewards = evaluate(agent, env, cfg.render, cfg.n_eval_episodes)
    print(f"Mean Rewards: {mean_rewards}, Std Rewards: {std_rewards}")

if __name__ == "__main__":
    main()