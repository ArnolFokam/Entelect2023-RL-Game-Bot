from ecbot.envs.spsa import SinglePlayerSingleAgentEnv, SinglePlayerSingleAgentEnvV2
from ecbot.envs.spsa_stacked_frames import SinglePlayerSingleAgentStackedFramesEnv, SinglePlayerSingleAgentStackedFramesEnvV2, SinglePlayerSingleAgentStackedFramesEnvV3


environments = {
    # flattened state
    "spsa": SinglePlayerSingleAgentEnv,
    
    # flattened state, ends on floor
    "spsa_v2": SinglePlayerSingleAgentEnvV2,
    
    # stacked observation, does not on floor
    "spsa_stacked": SinglePlayerSingleAgentStackedFramesEnv,
    
    # stacked observation, does not on floor, position in state
    "spsa_stacked_v2": SinglePlayerSingleAgentStackedFramesEnvV2,
    
    # stacked observation, ends on floor, position in state
    "spsa_stacked_v3": SinglePlayerSingleAgentStackedFramesEnvV3,
}