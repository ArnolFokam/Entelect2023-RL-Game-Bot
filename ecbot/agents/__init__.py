from ecbot.agents.dqn import DQN
from ecbot.agents.ppo import PPO
from ecbot.agents.ppo_lstm import PPO_LSTM


agents = {
    "dqn": DQN,
    "ppo": PPO,
    "ppo_lstm": PPO_LSTM,
}