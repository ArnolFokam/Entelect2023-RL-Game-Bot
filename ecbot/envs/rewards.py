from collections import defaultdict
from typing import Any

from ecbot.envs.cyfi import CyFi

class GetNewCoinsOnPlatform:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.last_collected = 0
        
        assert self.cfg.step_bad_reward <= 0.0
        assert self.cfg.on_floor_bad_reward < 0.0
        
        # to calculate reward
        self.position_reward = defaultdict(lambda : self.cfg.initial_position_reward)
        
    def __call__(self, info) -> Any:
        # negative step reward
        reward = self.cfg.step_bad_reward
        
        # reward for collection
        reward += (info["collected"] - self.last_collected) * self.cfg.coin_difference_reward_multiplier
        self.last_collected = info["collected"]
        
        # reward for not moving on the platform
        pos_x, pos_y = ((CyFi.window_width // 2) - 1), ((CyFi.window_height // 2))
        floor = info["window"][pos_y + 2:pos_y + 3, pos_x:pos_x + 2][0]
        
        if floor[0] == 1 or floor[1] == 1:
            reward += self.cfg.on_floor_bad_reward
            
        position = info["position"]
        reward += self.position_reward[position]
        
        # decay the reward
        self.position_reward[position] = reward * self.cfg.position_reward_decay_factor
        
        return reward
        
    
    def reset(self, info):
        self.last_collected = info["collected"]
        self.position_reward = defaultdict(lambda : self.cfg.initial_position_reward)


class NewPosition:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
        # to calculate reward
        self.position_reward = defaultdict(lambda : self.cfg.initial_position_reward)
    
    def __call__(self, info) -> float:
        position = info["position"]
        reward = self.position_reward[position]
        
        # decay the reward
        self.position_reward[position] = reward * self.cfg.position_reward_decay_factor
        
        return reward
    
    def reset(self, *args, **kwargs):
        self.position_reward = defaultdict(lambda : self.cfg.initial_position_reward)

reward_fn = {
    "new_position": NewPosition,
    "coins_on_platform": GetNewCoinsOnPlatform
}