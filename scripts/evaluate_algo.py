import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from box_env.envs import BoxEnv
from models.dqn import DQNAgent
from utils.save_load import load_checkpoint, load_config

os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

params = load_config('dqn.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = BoxEnv(render_mode=None)
env.action_space.seed(params['seed'])

def load_agent(agent_class, checkpoint_file, seed=None):
    agent = agent_class(
        state_shape=params['state_size'],
        action_size=params['action_size'],
        params=params,
        rng=np.random.default_rng(seed),
        device=device
    )
    agent.policy_net, agent.optimizer = load_checkpoint(
        agent.policy_net, agent.optimizer, checkpoint_file, device=agent.device
    )
    agent.update_target_network()
    agent.policy_net.eval()
    return agent

def mcts_action(agent, env, state, depth=5, n_sim=5, gamma=0.99):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]

    total_returns = np.zeros_like(q_values)
    for action in range(agent.action_size):
        returns = []
        for _ in range(n_sim):
            snapshot = env.clone_state()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            G = reward
            discount = gamma
            steps = 1
            while not done and steps < depth:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).permute(2,0,1).unsqueeze(0)
                with torch.no_grad():
                    q_next = agent.policy_net(obs_tensor)
                act = q_next.max(1)[1].item()
                obs, reward, terminated, truncated, _ = env.step(act)
                done = terminated or truncated
                G += discount * reward
                discount *= gamma
                steps += 1
            returns.append(G)
            env.restore_state(snapshot)
        total_returns[action] = np.mean(returns)
    return int(np.argmax(total_returns))

NUM_GAMES = 50
agent = load_agent(DQNAgent, "DQN.pth")
results = []

for game in range(NUM_GAMES):
    state, _ = env.reset(seed=params["seed"] + game)
    done = False
    cum_reward_dqn = 0
    steps_dqn = 0
    start_time = time.time()
    while not done:
        state_tensor = torch.from_numpy(state).float().permute(2,0,1).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
        action = q_values.max(1)[1].item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cum_reward_dqn += reward
        steps_dqn += 1
    end_time = time.time()
    avg_time_dqn = (end_time - start_time) / steps_dqn

    state, _ = env.reset(seed=params["seed"] + game)
    done = False
    cum_reward_mcts = 0
    steps_mcts = 0
    start_time = time.time()
    while not done:
        action = mcts_action(agent, env, state, depth=5, n_sim=5, gamma=params["gamma"])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cum_reward_mcts += reward
        steps_mcts += 1
    end_time = time.time()
    avg_time_mcts = (end_time - start_time) / steps_mcts

    results.append({
        "Game": game+1,
        "DQN_Reward": cum_reward_dqn,
        "DQN_Steps": steps_dqn,
        "DQN_AvgTime": avg_time_dqn,
        "MCTS_Reward": cum_reward_mcts,
        "MCTS_Steps": steps_mcts,
        "MCTS_AvgTime": avg_time_mcts
    })

    print(f"Game {game+1}: DQN Reward={cum_reward_dqn:.2f}, Steps={steps_dqn}, AvgTime={avg_time_dqn:.4f}s | "
          f"MCTS Reward={cum_reward_mcts:.2f}, Steps={steps_mcts}, AvgTime={avg_time_mcts:.4f}s")

df = pd.DataFrame(results)
df.to_csv("results/csv/Evaluation_Results.csv", index=False)
print("Results saved to results/csv/Evaluation_Results.csv")

plt.figure(figsize=(12,6))
plt.plot(df["Game"].to_numpy(), df["DQN_Reward"].to_numpy(), marker='o', label='DQN')
plt.plot(df["Game"].to_numpy(), df["MCTS_Reward"].to_numpy(), marker='x', label='MCTS-DQN')
plt.xlabel("Game Number")
plt.ylabel("Cumulative Reward")
plt.title(f"Cumulative Reward Comparison over {NUM_GAMES} Games")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/Evaluation_Results.png", dpi=300)
print("Plot saved to results/plots/Evaluation_Results.png")
plt.show()
