from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from ecbot.envs.single_player_single_agent import SinglePlayerSingleAgentEnv

if __name__ == "__main__":
    # Use a separate environement for evaluation
    eval_env = SinglePlayerSingleAgentEnv(render_mode="human")
    
    # Load the trained agent
    model = DQN.load("dqn_single_player_single_agent")

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env,
        render=True,
        n_eval_episodes=100
    )

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
