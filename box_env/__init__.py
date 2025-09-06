from gymnasium.envs.registration import register

register(
    id="box_env/BoxEnv-v0",
    entry_point="box_env.envs:BoxEnv",
)
