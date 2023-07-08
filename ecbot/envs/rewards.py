from collections import defaultdict
from typing import Any

from ecbot.envs.cyfi import CyFi

class GetNewCoinsOnPlatform:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.last_collected = 0
        
        assert self.cfg.step_penalty <= 0.0
        assert self.cfg.on_bad_floor_penalty <= 0.0
        assert self.cfg.on_good_floor_reward >= 0.0
        
        # to calculate reward
        self.position_reward = defaultdict(lambda : self.cfg.initial_position_reward)
        
    def __call__(self, info) -> Any:
        # negative step reward
        reward = self.cfg.step_penalty
        
        was_on_bad_floor = False
        
        # -/+ ve reward for collection or lost
        coin_difference = info["collected"] - self.last_collected
        
        if coin_difference > 0:
            reward += coin_difference * self.cfg.more_coin_reward_multiplier
        else:
            reward += coin_difference * self.cfg.less_coin_penalty_multiplier
            
        self.last_collected = info["collected"]
        
        # get floor position of agent
        pos_x, pos_y = ((CyFi.window_width // 2) - 1), ((CyFi.window_height // 2))
        floor = info["window"][pos_y + 2:pos_y + 3, pos_x:pos_x + 2][0]
        
        # penalize for being on a bad floor
        if floor[0] in self.cfg.bad_floors or floor[1] in self.cfg.bad_floors:
            reward += self.cfg.on_bad_floor_penalty
            was_on_bad_floor = True
        
        # reward for being on a good floor
        if floor[0] in self.cfg.good_floors or floor[1] in self.cfg.good_floors:
            reward += self.cfg.on_good_floor_reward
            
        position = info["position"]
        reward += self.position_reward[position]
        
        # decay the reward
        self.position_reward[position] = reward * self.cfg.position_reward_decay_factor
        
        return reward, was_on_bad_floor
        
    
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