import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils.save_load import load_config, save_csv, save_checkpoint
from utils.plotting import plot_rewards_per_episode
from utils.helper import postprocess

from models.dqn import DQNAgent

from box_env.envs.box_env import BoxEnv

params = load_config('dqn.yaml')
rng = np.random.default_rng(params['seed'])

def run_env(env, params):
    rewards = np.zeros((params['total_episodes'], params['n_runs']))
    steps = np.zeros((params['total_episodes'], params['n_runs']))
    episodes = np.arange(params['total_episodes'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in range(params['n_runs']):
        agent = DQNAgent(
            state_shape=params['state_size'],
            action_size=params['action_size'],
            params=params,
            rng=rng,
            device=device
        )
        for episode in tqdm(range(params['total_episodes']), desc=f"Run {run+1}"):
            state, _ = env.reset(seed=params['seed'] + run)
            done = False
            total_rewards = 0
            step = 0
            exploration_rate = params['epsilon_end'] + (params['epsilon_start'] - params['epsilon_end']) * np.exp(-1. * episode / params['epsilon_decay'])
            while not done:
                action = agent.choose_action(state, exploration_rate)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_rewards += reward
                step += 1
                agent.train_step(params['batch_size'])
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
            if episode % params['target_update_freq'] == 0:
                agent.update_target_network()
    save_checkpoint(agent.policy_net, agent.optimizer, "DQN.pth")
    return rewards, steps, episodes

if __name__ == '__main__':
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    env = BoxEnv(render_mode=None)

    env.action_space.seed(params['seed'])

    print("\n" + "="*50)
    print("TRAINING PARAMS")
    print("-"*50)
    print(f"Environment size       : {params['map_size']}x{params['map_size']}")
    print(f"Total episodes         : {params['total_episodes']}")
    print(f"Number of runs         : {params['n_runs']}")
    print(f"Action space size      : {params['action_size']}")
    print(f"State space shape      : {params['state_size']}")
    print(f"Learning rate          : {params['learning_rate']}")
    print(f"Discount factor (gamma): {params['gamma']}")
    print(f"Epsilon start          : {params['epsilon_start']}")
    print(f"Epsilon end            : {params['epsilon_end']}")
    print(f"Epsilon decay          : {params['epsilon_decay']}")
    print(f"Batch size             : {params['batch_size']}")
    print(f"Max memory             : {params['max_memory']}")
    print(f"Target update freq     : {params['target_update_freq']}")
    print(f"Seed                   : {params['seed']}")
    print("="*50 + "\n")
    print(f"Training on BoxEnv of size: {params['map_size']}x{params['map_size']} started!\n")

    rewards, steps, episodes = run_env(env, params)
    
    res, st = postprocess(episodes, params, rewards, steps)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    save_csv(res_all, "DQN.csv")

    plot_rewards_per_episode(res_all, 'DQN')

    env.close()
