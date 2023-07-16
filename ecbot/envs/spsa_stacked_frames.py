import numpy as np
from collections import deque

import gym
from gym import spaces

from ecbot.connection import Constants
from ecbot.envs.cyfi import CyFi
from ecbot.envs.rewards import reward_fn

class SinglePlayerSingleAgentStackedFramesEnv(CyFi):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # Our CiFy agent has 12 possible actions
        # UP - 1
        # DOWN- 2
        # LEFT - 3
        # RIGHT - 4
        # UPLEFT - 5
        # UPRIGHT - 6
        # DOWNLEFT - 7
        # DOWNRIGHT - 8
        # DIGDOWN - 9
        # DIGLEFT - 10
        # DIGRIGHT - 11
        # STEAL - 12 (WIP wrench)
        self.action_space = gym.spaces.Discrete(12)
        
        # Our CiFy agent only knows about its windowed view of the world
        # Note: the agent is at the center of the window
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.cfg.num_frames, 34, 22), dtype=float)
        
        self.reward_fn = reward_fn[self.cfg.reward_fn](self.cfg)
        self.past_k_rewards = deque([], maxlen=self.cfg.reward_backup_len)
        
        self.game_has_reset = False
        
    def reset(self):
        self.game_has_reset = True
        
        observation = super().reset()
        self.past_k_rewards = deque([], maxlen=self.cfg.reward_backup_len)
        self.reward_fn.reset(self.info)
        
        return observation
    
    def step(self, action: int):
        self.game_has_reset = False
        
        # command from the  game server are 1-indexed
        self.game_client.send_player_command(int(action + 1))
        self._wait_for_game_state()
        
        self.observation, self.info, done = self._return_env_state()
        
        reward, _ = self.reward_fn(self.info, )
        
        self.past_k_rewards.append(reward)
        print(f"reward: {reward}, mean: {np.mean(self.past_k_rewards)}")
        done = done or np.mean(self.past_k_rewards) < self.cfg.past_reward_threshold
        
        return self.observation, reward, done, self.info
        
    def _get_observation(self, game_state):
        if self.game_has_reset:
            # use the start frame in all frames
            return np.array([game_state[Constants.HERO_WINDOW]] * self.cfg.num_frames, dtype=np.float32) / 6.0
        else:
            # enqueue the current frame and dequeue the oldest frame
            return np.concatenate([
                self.observation[1:], [
                    np.array(game_state[Constants.HERO_WINDOW], dtype=np.float32) / 6.0
                ]], axis=0)
        
    def _get_info(self, game_state):
        return {
            "position": (
                game_state[Constants.POSITION_X],
                game_state[Constants.POSITION_Y],
            ),
            "window": np.rot90(game_state[Constants.HERO_WINDOW]),
            "elapsed_time": game_state[Constants.ELAPSED_TIME],
            "collected": game_state[Constants.COLLECTED],
            "current_level": game_state[Constants.CURRENT_LEVEL]
        }
        
class SinglePlayerSingleAgentStackedFramesEnvV2(SinglePlayerSingleAgentStackedFramesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.cfg.num_frames + 2, 34, 22), dtype=float)
        
    def _get_observation(self, game_state):
        if self.game_has_reset:
            # use the start frame in all frames
            window = np.array([game_state[Constants.HERO_WINDOW]] * self.cfg.num_frames, dtype=np.float32) / 6.0
            return np.concatenate([
                [np.full_like(window[0], fill_value=game_state[Constants.POSITION_X]) / 1000.0],
                [np.full_like(window[0], fill_value=game_state[Constants.POSITION_Y]) / 1000.0],
                window
            ], axis=0)
        else:
            # enqueue the current frame and dequeue the oldest frame
            return np.concatenate([
                [np.full_like(self.observation[0], fill_value=game_state[Constants.POSITION_X]) / 1000.0],
                [np.full_like(self.observation[0], fill_value=game_state[Constants.POSITION_Y]) / 1000.0],
                self.observation[3:], 
                [np.array(game_state[Constants.HERO_WINDOW], dtype=np.float32) / 6.0]
            ], axis=0)