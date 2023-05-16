from ecbot.agents.base import BaseAgent
from ecbot.agents.policies import policies

class PPO(BaseAgent):
    
    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        
        self.model = PPO(
            policies[cfg.poliy_name], 
            env,
            verbose=0,
            n_epochs=cfg.num_epochs,
            gamma=cfg.gamma,
            n_steps=cfg.num_steps,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
        )
        
    