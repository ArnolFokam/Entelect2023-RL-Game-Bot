import numpy as np

from stable_baselines3 import DQN

from .envs.single_player_single_agent import SinglePlayerSingleAgentEnv

if __name__ == "__main__":
    # there is only one player and one 
    # learnig agent in this environment
    env = SinglePlayerSingleAgentEnv()
    episodes = 10
    
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000, progress_bar=True)
    
    # save agent
    model.save("dqn_single_player_single_agent")
    
    # only relavent when we need to visualise the agent
    
    # all_episode_rewards = []
    
    # for i in range(episodes):
        
    #     episode_reward = []
        
    #     print("Episode {} started".format(i))
        
    #     done = False
    #     observation = env.reset()
        
    #     while not done:
    #         action, _ = model.predict(observation, deterministic=False)
    #         # print("Action:", action)
    #         # action = env.action_space.sample()  # choose random action
    #         observation, reward, done, info = env.step(action)  # feedback from environment
    #         episode_reward.append(reward) # save reward for this step
    #         env.render(info) # render environment
            
    #     all_episode_rewards.append(sum(episode_reward))
        
    #     mean_episode_reward = np.mean(all_episode_rewards)
    #     print("Episode {} finished".format(i))
    #     print("Mean reward:", mean_episode_reward, "Num episodes:", episodes)
        
        