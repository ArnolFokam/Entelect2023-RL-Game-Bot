from ecbot.envs.single_player_single_agent import SinglePlayerSingleAgentEnv, SinglePlayerSingleAgentEnvV2, SinglePlayerSingleAgentEnvV3


environments = {
    "spsa": SinglePlayerSingleAgentEnv,
    "spsa_v2": SinglePlayerSingleAgentEnvV2,
    "spsa_v3": SinglePlayerSingleAgentEnvV3
}