from ecbot.envs.spsa import SinglePlayerSingleAgentEnv, SinglePlayerSingleAgentEnvV2
from ecbot.envs.spsa_stacked_frames import SinglePlayerSingleAgentStackedFramesEnv


environments = {
    "spsa": SinglePlayerSingleAgentEnv,
    "spsa_v2": SinglePlayerSingleAgentEnvV2,
    "spsa_stacked": SinglePlayerSingleAgentStackedFramesEnv,
}