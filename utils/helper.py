import numpy as np
import pandas as pd
import time
from box_env.envs.box_env import BoxEnv, Actions

def postprocess(episodes, params, rewards, steps):
    res = []
    for run in range(params['n_runs']):
        df = pd.DataFrame({
            "Episodes": episodes,
            "Rewards": rewards[:, run],
            "Steps": steps[:, run],
            "cum_rewards": np.cumsum(rewards[:, run]),
            "Run": run + 1,
        })
        res.append(df)
    res = pd.concat(res, ignore_index=True)

    st = pd.DataFrame({
        "Episodes": episodes,
        "Steps": steps.mean(axis=1),
    })
    return res, st

def get_discrete_state(env):
    x, y = env._agent_location
    return np.ravel_multi_index((x, y), (env.size, env.size))

def test_env(render=True, sleep_time=0.1):
    env = BoxEnv(render_mode="human" if render else None)
    observation, info = env.reset()
    print("Initial state observation shape:", observation.shape)
    print("Initial info:", info)

    terminated, truncated, steps = False, False, 0
    print("\nStarting simulation...")

    while not terminated and not truncated:
        action = env.action_space.sample()
        print(f"\nStep {steps}: Action chosen: {Actions(action).name}")

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")

        time.sleep(sleep_time)
        steps += 1

    env.close()
    print("\nSimulation finished.")
