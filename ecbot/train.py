# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

from .envs.single_player_single_agent import SinglePlayerSingleAgentEnv

if __name__ == "__main__":
    # there is only one player and one 
    # learnig agent in this environment
    env = SinglePlayerSingleAgentEnv(render_mode="human")
    episodes = 10
    
    # model = DQN(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=25000)
    
    for i in range(episodes):
        
        print("Episode {} started".format(i))
        
        done = False
        observation = env.reset()
        
        while not done:
            action = env.action_space.sample()  # choose random action
            observation, reward, done, info = env.step(action)  # feedback from environment
            env.render(observation)  # render the environment

        print("Episode {} finished".format(i))