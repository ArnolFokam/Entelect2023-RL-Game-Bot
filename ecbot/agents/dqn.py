import os
import math
import random
from copy import deepcopy
from itertools import count
from collections import deque, namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from ecbot.agents.base import BaseAgent
from ecbot.agents.function_approximators import function_approximators


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Replay memory to sample transitions"""
    
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(BaseAgent):
    """Deep-Q Network agent"""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.steps_done = 0
        
        self.policy_network = function_approximators[self.cfg.q_approximator](
            input_shape=self.observation_shape,
            output_shape=(self.num_actions,), 
            hidden_dim=self.cfg.q_hidden_dim,
            num_hidden_layers=self.cfg.q_num_hidden_layers
        )
        self.target_network = deepcopy(self.policy_network)
        
    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * math.exp(-1. * self.steps_done / self.cfg.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.act(state)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
    def act(self, state):
        return self.policy_network(state).max(1)[1].view(1, 1)
    
    def learn(self):
        self.target_network = self.target_network.to(self.device)
        self.policy_network = self.policy_network.to(self.device)      
        optimizer = optim.AdamW(self.policy_network.parameters(), lr=self.cfg.learning_rate, amsgrad=True)
        memory = ReplayMemory(self.cfg.buffer_size)
        
        for i_episode in range(self.cfg.num_episodes):
            
            print(f"Starting {i_episode + 1}th episode")
            
            state, done = self.env.reset(), False
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            ep_rewards = 0.0
            
            for _ in range(self.cfg.max_train_steps):
                action = self._select_action(state)
                observation, reward, done, truncated, _ = self.env.step(action.item())
                done = done or truncated
                reward = torch.tensor([reward], device=self.device)
                
                # training viz
                frame = self.env.render()
                self.env.send_frame_to_socket_server(frame)
                
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                # store the transition in mermory
                memory.push(state, action, next_state, reward)
                
                # move to the next state
                state = next_state
                ep_rewards  += reward
                
                # perform one step of the optimization (on the policy network)
                if len(memory) >= self.cfg.batch_size:
                    transitions = memory.sample(self.cfg.batch_size)
                    batch = Transition(*zip(*transitions))
                    
                    # Compute a mask of non-final states and concatenate the batch elements
                    # (a final state would've been the one after which simulation ended)
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = self.policy_network(state_batch).gather(1, action_batch)
                    
                    # Compute V(s_{t+1}) for all next states.
                    # Expected values of actions for non_final_next_states are computed based
                    # on the "older" target_net; selecting their best reward with max(1)[0].
                    # This is merged based on the mask, such that we'll have either the expected
                    # state value or 0 in case the state was final.
                    next_state_values = torch.zeros(self.cfg.batch_size, device=self.device)
                    with torch.no_grad():
                        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
                        
                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * self.cfg.gamma) + reward_batch
                    
                    # Compute Huber loss
                    criterion = nn.SmoothL1Loss()
                    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                    
                     # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # In-place gradient clipping
                    torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
                    optimizer.step()
                    
                self.steps_done += 1
                
                if done:
                    break
                    
                
                # soft update of the q-network's weeights
                target_network_weights = self.target_network.state_dict()
                policy_network_weights = self.policy_network.state_dict()
                for k in policy_network_weights:
                    target_network_weights[k] = policy_network_weights[k]*self.cfg.tau + target_network_weights[k]*(1 - self.cfg.tau)
                self.target_network.load_state_dict(target_network_weights)
            
            logs={}
            if (i_episode + 1) % self.cfg.log_every_episodes == 0:
                logs["train-episode-reward"] = ep_rewards
                
            if (i_episode + 1) % self.cfg.eval_every_episodes == 0:
                mean_rwd, _, min_bv_frames, max_bv_frames = self.evaluate(return_frames=True)
                min_bv_frames = np.rollaxis(min_bv_frames, -1, 1)
                max_bv_frames = np.rollaxis(max_bv_frames, -1, 1)
                
                logs["eval-mean-reward"] = mean_rwd
                logs["eval-min-behaviour"] = wandb.Video(min_bv_frames, fps=self.env.metadata["render_fps"])
                logs["eval-max-behaviour"] = wandb.Video(max_bv_frames, fps=self.env.metadata["render_fps"])
                
            if len(logs) > 0:
                wandb.log({
                    **logs,
                    "episode": i_episode + 1
                }, step=self.steps_done)
            
    def save(self, dir):
        torch.save({
            "target_network": self.target_network.state_dict(),
            "cfg": self.cfg
        }, os.path.join(dir, "dqn.pt"))
        
    @classmethod
    def load_trained_agent(cls, cfg, dir, env):
        # load the agent class
        agent = cls(cfg, env)
        
        # load the network weights
        artifacts = torch.load(os.path.join(dir, "dqn.pt"))
        agent.policy_network.load_state_dict(artifacts["target_network"], map_location=torch.device('cpu'))
        
        return agent
        
    
        
            
            
        
        
        
        
    