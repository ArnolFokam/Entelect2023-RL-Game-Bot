from stable_baselines3 import PPO

from .envs.single_player_single_agent import SinglePlayerSingleAgentEnv

if __name__ == "__main__":
    gamma=0.99
    batch_size=64
    num_epochs=10
    num_steps=2048
    learning_rate=0.00003
    training_steps=220000
    
    # there is only one player and one 
    # learnig agent in this environment
    env = SinglePlayerSingleAgentEnv()
    
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=0,
        n_epochs=num_epochs,
        gamma=gamma,
        n_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    model.learn(
        total_timesteps=training_steps, 
        progress_bar=True
    )
    
    # save agent
    model.save("dqn_single_player_single_agent")
        
        