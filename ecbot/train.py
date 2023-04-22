from .environment import CyFiEnv

if __name__ == "__main__":
    env = CyFiEnv()
    observation = env.reset()
    done = False
    t = 0
    
    while not done:
        action = env.action_space.sample()  # choose random action
        observation, reward, done, info = env.step(action)  # feedback from environment
        t += 1
        if not t % 100:
            print(t, info)