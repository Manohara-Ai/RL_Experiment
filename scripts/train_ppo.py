import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils.save_load import load_config, save_csv
from utils.plotting import plot_rewards_per_episode
from utils.helper import postprocess

from models.ppo import PPOAgent
from box_env.envs.box_env import BoxEnv

params = load_config('ppo.yaml')
rng = np.random.default_rng(params['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_env(env, params):
    rewards = np.zeros((params['total_episodes'], params['n_runs']))
    steps = np.zeros((params['total_episodes'], params['n_runs']))
    episodes = np.arange(params['total_episodes'])

    N = params.get('N')
    best_score = -float('inf')

    for run in range(params['n_runs']):
        agent = PPOAgent(
            n_actions=params['action_size'],
            input_dims=params['state_size'],
            gamma=params['gamma'],
            lr=params['learning_rate'],
            gae_lambda=params.get('gae_lambda'),
            policy_clip=params.get('policy_clip'),
            batch_size=params.get('batch_size'),
            n_epochs=params.get('n_epochs')
        )

        n_steps = 0
        for episode in tqdm(range(params['total_episodes']), desc=f"Run {run+1}"):
            state, _ = env.reset(seed=params['seed'] + run)
            state = np.array(state, dtype=np.float32)
            done = False
            total_rewards = 0
            step = 0

            while not done:
                action, prob, val = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_rewards += reward

                agent.remember(state, action, prob, val, reward, done)
                state = np.array(next_state, dtype=np.float32)

                if n_steps % N == 0:
                    agent.learn()

                n_steps += 1
                step += 1

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

            avg_score_run = rewards[max(0, episode-100):episode+1, run].mean()
            
            if avg_score_run > best_score:
                best_score = avg_score_run
                agent.save_models()

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
    print(f"Batch size             : {params['batch_size']}")
    print(f"Number of epochs       : {params['n_epochs']}")
    print(f"Learning frequency N   : {params['N']}")
    print(f"Learning rate          : {params['learning_rate']}")
    print(f"Discount factor (gamma): {params['gamma']}")
    print(f"GAE lambda             : {params['gae_lambda']}")
    print(f"Policy clip            : {params['policy_clip']}")
    print(f"Input dimensions       : {params['state_size']}")
    print(f"Number of actions      : {params['n_actions']}")
    print(f"Seed                   : {params['seed']}")
    print("="*50 + "\n")
    print(f"Training on BoxEnv of size: {params['map_size']}x{params['map_size']} started!\n")

    rewards, steps, episodes = run_env(env, params)
    
    res, st = postprocess(episodes, params, rewards, steps)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    save_csv(res_all, "PPO.csv")
    print("Checkpoint saved at:", os.path.abspath("results/checkpoints/PPO/"))
    
    plot_rewards_per_episode(res_all, 'PPO')
    
    env.close()
