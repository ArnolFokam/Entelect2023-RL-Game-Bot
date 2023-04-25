from stable_baselines3 import DQN

from .envs.single_player_single_agent import SinglePlayerSingleAgentEnv

if __name__ == "__main__":
    # there is only one player and one 
    # learnig agent in this environment
    env = SinglePlayerSingleAgentEnv()
    
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000, progress_bar=True)
    
    # save agent
    model.save("dqn_single_player_single_agent")
        
        