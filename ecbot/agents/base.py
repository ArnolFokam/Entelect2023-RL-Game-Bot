import numpy as np
from torch import Tensor
import torch


class BaseAgent:
    """A wrapper module for our learning agent."""
    
    def __init__(self, cfg, env, *args, **kwargs) -> None:
        self.cfg = cfg
        self.num_actions = env.action_space.n
        
        # initialize the env and info about the env
        self.env = env
        self.num_actions = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape
        
        # other agent things
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        
    def learn(self):
        raise NotImplementedError
        
    def save(self):
        raise NotImplementedError
        
    def load(self):
        raise NotImplementedError
    
    def evaluate(self, return_frames=False):
        
        episodes_rewards = []
        
        if return_frames:
            episodes_frames = []
        
        for _ in range(self.cfg.n_eval_episodes):
            done = False
            episode_reward = 0.0
            
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            frames = []
            
            for _ in range(self.cfg.max_eval_steps):
                action = self.act(state)
                state, reward, done, _, _ = self.env.step(action)
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                episode_reward += reward
                
                if return_frames:
                    frame = self.env.render()
                    frames.append(frame)
                    
                if done:
                    break
            
            # add the reward to the episodes
            episodes_rewards.append(episode_reward)
            
            if return_frames:
                frames = np.array(frames)
                episodes_frames.append(frames)
                
            
        episodes_rewards = np.array(episodes_rewards)
        
        if return_frames:
            min_behaviour_frame = episodes_frames[np.argmin(episodes_rewards)]
            max_behaviour_frame = episodes_frames[np.argmax(episodes_rewards)]
            
        return np.mean(episodes_rewards), np.std(episodes_rewards), min_behaviour_frame, max_behaviour_frame