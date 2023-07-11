import os
import numpy as np

import torch
import torch.optim as optim
import wandb
from torch.distributions.categorical import Categorical

from ecbot.agents.function_approximators import function_approximators
from ecbot.agents.base import BaseAgent


class PPO(BaseAgent):
    """Proximal Policy Optimization"""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.steps_done = 0
        
        self.actor_network = function_approximators[self.cfg.actor_approximator](
            input_shape=self.observation_shape,
            output_shape=(self.num_actions,), 
            hidden_dim=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.num_hidden_layers
        )
        
        self.critic_network = function_approximators[self.cfg.critic_approximator](
            input_shape=self.observation_shape,
            output_shape=(1,), 
            hidden_dim=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.num_hidden_layers
        )
        
    def _select_action(self, state):
        action_logits = self.actor_network(state)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action.item(), action_log_prob.item()
        
        
    def act(self, state):
        action, _ = self._select_action(state)
        return action
    
    def compute_general_advantage_estimates(self, rewards, values):
        next_values = np.concatenate([values[1:], [0]])
        deltas = [rew + self.cfg.gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + self.cfg.decay * self.cfg.gamma * gaes[-1])

        return np.array(gaes[::-1])
    
    def compute_returns(self, rewards):
        returns = [float(rewards[-1])]
        for i in reversed(range(len(rewards) - 1)):
            returns.append(float(rewards[i]) + self.cfg.gamma * returns[-1])
        return np.array(returns[::-1])
    
    def learn(self):
        # to device
        self.actor_network = self.actor_network.to(self.device)
        self.critic_network = self.critic_network.to(self.device) 
        
        # optimizers     
        actor_optimizer = optim.AdamW(self.actor_network.parameters(), lr=self.cfg.actor_learning_rate, amsgrad=True)
        critic_optimizer = optim.AdamW(self.critic_network.parameters(), lr=self.cfg.critic_learning_rate, amsgrad=True)
        
        for i_episode in range(self.cfg.num_episodes):
            
            print(f"Starting {i_episode + 1}th episode")
            
            # perform a rollout
            state = self.env.reset()
            ep_rewards = 0.0
            
            states = []
            actions = []
            action_probs = []
            values = []
            rewards = []
            
            # collect transitions
            for _ in range(self.cfg.max_train_steps):
                obs = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
                
                # get the  action and value for that state
                action, action_log_prob = self._select_action(obs)
                value = self.critic_network(obs).item()
                
                next_state, reward, done, _ = self.env.step(action)
                
                # training viz
                frame = self.env.render()
                self.env.send_frame_to_socket_server(frame)
                
                # append interesting values
                states.append(state)
                actions.append(action)
                values.append(value)
                rewards.append(reward)
                action_probs.append(action_log_prob)
                
                # move to the next state
                state = next_state
                ep_rewards += reward
                
                self.steps_done += 1
                
                if done:
                    break
            
            # calculate gaes and returns
            advantage_estimates = self.compute_general_advantage_estimates(rewards, values)
            target_values = self.compute_returns(rewards)
            
            # shuffle the data
            permute_idxs = np.random.permutation(len(target_values))
            
            # get the required transitions in order
            states = torch.tensor(np.asarray(states)[permute_idxs], dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.asarray(actions)[permute_idxs], dtype=torch.int32, device=self.device)
            old_actions_log_prob = torch.tensor(np.asarray(action_probs)[permute_idxs], dtype=torch.float32, device=self.device)
            target_values = torch.tensor(np.asarray(target_values)[permute_idxs], dtype=torch.float32, device=self.device)
            advantage_estimates = torch.tensor(np.asarray(advantage_estimates), dtype=torch.float32, device=self.device)
            
            # optimize actor
            actor_loss = []
            for _ in range(self.cfg.max_actor_train_iterations):
                
                new_actions_logits = self.actor_network(states)
                action_dist = Categorical(logits=new_actions_logits)
                new_actions_log_prob = action_dist.log_prob(actions)

                # policy ratio
                policy_ratio = torch.exp(new_actions_log_prob - old_actions_log_prob)
                clipped_ratio = policy_ratio.clamp(
                    1 - self.cfg.ppo_clip_val, 1 + self.cfg.ppo_clip_val
                )

                # clipped loss
                loss = -torch.min(policy_ratio * advantage_estimates,  clipped_ratio * advantage_estimates).mean()
                actor_loss.append(loss.item())
                
                # backpop through actor
                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                
                # check if we changed the policy enough?
                if torch.log(policy_ratio).mean() >= self.cfg.target_kl_divergence:
                    break
                
            # optimize critic
            critic_loss = []
            for _ in range(self.cfg.critic_train_iterations):
                
                pred_values = self.critic_network(states)
                loss = ((target_values - pred_values) ** 2).mean()
                critic_loss.append(loss.item())
                
                # backpop through critic
                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()
    
            
            logs={}
            if (i_episode + 1) % self.cfg.log_every_episodes == 0:
                logs["train-episode-reward"] = ep_rewards
                logs["train-actor-loss"] = np.mean(actor_loss)
                logs["train-critic-loss"] = np.mean(critic_loss)
                
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
                
        self.env.disconnect_socket_server()
            
    def save(self, dir):
        torch.save({
            "actor_network": self.actor_network.state_dict(),
            "critic_network": self.critic_network.state_dict(),
            "cfg": self.cfg
        }, os.path.join(dir, "critic.pt"))
        
            
            
        
        
        
        
    