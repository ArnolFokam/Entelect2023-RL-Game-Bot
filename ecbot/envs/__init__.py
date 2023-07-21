from ecbot.envs.spsa import SinglePlayerSingleAgentEnv, SinglePlayerSingleAgentEnvV2
from ecbot.envs.spsa_stacked_frames import SinglePlayerSingleAgentStackedFramesEnvV4, SinglePlayerSingleAgentStackedFramesEnvV5, SinglePlayerSingleAgentStackedFramesEnv, SinglePlayerSingleAgentStackedFramesEnvV2, SinglePlayerSingleAgentStackedFramesEnvV3


environments = {
    # flattened state
    "spsa": SinglePlayerSingleAgentEnv,
    
    # flattened state
    "spsa_v2": SinglePlayerSingleAgentEnvV2,
    
    # stacked observation
    "spsa_stacked": SinglePlayerSingleAgentStackedFramesEnv,
    
    # stacked observation, position in state
    "spsa_stacked_v2": SinglePlayerSingleAgentStackedFramesEnvV2,
    
    # [WARNING: this is no more supported] stacked observation, position in state 
    "spsa_stacked_v3": SinglePlayerSingleAgentStackedFramesEnvV3,
    
    # stacked observation, reduced action space
    "spsa_stacked_v4": SinglePlayerSingleAgentStackedFramesEnvV4,
    
    # stacked observation, reduced action space + one no action
    "spsa_stacked_v5": SinglePlayerSingleAgentStackedFramesEnvV5,
}