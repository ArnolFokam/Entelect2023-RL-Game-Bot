from ecbot.envs.spsa import SinglePlayerSingleAgentEnv, SinglePlayerSingleAgentEnvV2
from ecbot.envs.spsa_stacked_frames import SinglePlayerSingleAgentStackedFramesEnv, SinglePlayerSingleAgentStackedFramesEnvV2, SinglePlayerSingleAgentStackedFramesEnvV3


environments = {
    "spsa": SinglePlayerSingleAgentEnv,
    "spsa_v2": SinglePlayerSingleAgentEnvV2,
    "spsa_stacked": SinglePlayerSingleAgentStackedFramesEnv,
    "spsa_stacked_v2": SinglePlayerSingleAgentStackedFramesEnvV2,
    "spsa_stacked_v3": SinglePlayerSingleAgentStackedFramesEnvV3,
}