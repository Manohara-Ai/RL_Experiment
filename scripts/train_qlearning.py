import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.save_load import load_config, save_csv, save_qtable
from utils.plotting import plot_rewards_per_episode
from utils.helper import postprocess, get_discrete_state

from models.qlearning import Qlearning, EpsilonGreedy

from box_env.envs.box_env import BoxEnv

params = load_config('qlearning.yaml')
rng = np.random.default_rng(params['seed'])

def run_env(env, params):
    rewards = np.zeros((params['total_episodes'], params['n_runs']))
    steps = np.zeros((params['total_episodes'], params['n_runs']))
    episodes = np.arange(params['total_episodes'])
    qtables = np.zeros((params['n_runs'], params['state_size'], params['action_size']))
    all_states, all_actions = [], []

    best_avg_reward = -float('inf')
    best_qtable = None

    for run in range(params['n_runs']):
        learner = Qlearning(
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            state_size=params['state_size'],
            action_size=params['action_size']
        )
        explorer = EpsilonGreedy(params['epsilon'], rng)

        for episode in tqdm(range(params['total_episodes']), desc=f"Run {run+1}"):
            env.reset(seed=params['seed'] + run)
            state = get_discrete_state(env)

            done = False
            total_rewards = 0
            step = 0

            while not done:
                action = explorer.choose_action(env.action_space, state, learner.qtable)

                all_states.append(state)
                all_actions.append(action)

                _, reward, terminated, truncated, _ = env.step(action)
                new_state = get_discrete_state(env)

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1
                state = new_state
                done = terminated or truncated

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

            avg_reward_run = rewards[max(0, episode-100):episode+1, run].mean()
            
            if avg_reward_run > best_avg_reward:
                best_avg_reward = avg_reward_run
                best_qtable = learner.qtable.copy()

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions, best_qtable

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
    print(f"State space size       : {params['state_size']}")
    print(f"Learning rate          : {params['learning_rate']}")
    print(f"Discount factor (gamma): {params['gamma']}")
    print(f"Epsilon (exploration)  : {params['epsilon']}")
    print(f"Seed                   : {params['seed']}")
    print("="*50 + "\n")
    print(f"Training on BoxEnv of size: {params['map_size']}x{params['map_size']} started!\n")
    
    rewards, steps, episodes, qtables, all_states, all_actions = run_env(env, params)

    res, st = postprocess(episodes, params, rewards, steps)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)
    save_qtable(qtable, 'Q_Table.npy')
    save_csv(res_all, "Q_Learning.csv")

    plot_rewards_per_episode(res_all, "Q_Learning")

    env.close()
