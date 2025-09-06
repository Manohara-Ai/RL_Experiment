# RL\_Experiment

A reinforcement learning project exploring different RL algorithms on a custom-built grid-world environment (**BoxEnv**).
Currently implemented: **Q-Learning** and **DQN**.
Future extensions include **TreeQN** and **MCTS guided by DQN**.

---

## Project Overview

![Box Environment](box_env/envs/box_env.png)

Train agents to navigate and solve the `BoxEnv` environment using different reinforcement learning techniques.

---

## Project Structure

```
.
├── box_env
│   ├── envs
│   │   ├── box_env.py        # Custom environment implementation
│   │   └── resources/        # Textures (player, goal, walls, obstacles)
│   └── wrappers/             # Optional wrappers for preprocessing
│
├── config
│   ├── dqn.yaml              # Hyperparameters for DQN
│   └── qlearning.yaml        # Hyperparameters for Q-Learning
│
├── models
│   ├── dqn.py                # DQN agent implementation
│   └── qlearning.py          # Q-Learning agent implementation
│
├── scripts
│   ├── train_dqn.py          # Training script for DQN
│   └── train_qlearning.py    # Training script for Q-Learning
│
├── results
│   ├── checkpoints/          # Saved models and Q-tables
│   ├── csv/                  # Training logs in CSV format
│   └── plots/                # Training performance plots
│
├── utils
│   ├── helper.py             # Helper functions
│   ├── plotting.py           # Plotting utilities
│   └── save_load.py          # Save/load utilities
│
├── main.py                   # CLI entrypoint for experiments
├── pyproject.toml            # Project metadata
├── LICENSE
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Manohara-Ai/RL_Experiment.git
cd RL_Experiment
```

2. Install dependencies:

```bash
pip install -e .
```

---

## Running Experiments

Use the `main.py` entrypoint to run training with the algorithm of your choice:

### Run Q-Learning

```bash
python3 main.py --algo qlearning
```

### Run DQN

```bash
python3 main.py --algo dqn
```

This will:

* Load hyperparameters from the corresponding YAML config in `config/`
* Initialize the **BoxEnv** environment
* Train the chosen agent
* Save results in `results/` (checkpoints, logs, and plots)

---

## Results

Training outputs are automatically saved in the `results/` directory:

* **Plots** → `results/plots/`
* **Logs (CSV)** → `results/csv/`
* **Checkpoints** → `results/checkpoints/`

---

### Q-Learning Performance

The Q-Learning agent's performance appears to be stagnant. The average reward fluctuates around a negative value and does not show a clear upward trend. This suggests that the agent is struggling to find a consistent optimal policy. This behavior is expected for tabular Q-learning in an environment with a large state space, as it may not be able to explore all possible states effectively. The agent seems to get stuck in local optima, receiving a consistent negative reward, likely from bumping into walls or obstacles without reaching the goal.

[![Q-Learning Plot](results/plots/Q_Learning.png)](results/plots/Q_Learning.png)

---

### DQN Performance

In contrast, the DQN agent demonstrates significant learning progress. The average reward steadily increases over the 2000 episodes, starting from a negative value and climbing to a positive one. This indicates that the deep neural network is successfully generalizing from the agent's experiences to learn a better policy. The smoother curve and consistent upward trend show that DQN is more effective at handling the complexity of the BoxEnv and navigating to the goal, unlike the tabular Q-Learning approach. The final positive reward shows that the agent has learned a policy to solve the environment.

[![DQN Plot](results/plots/DQN.png)](results/plots/DQN.png)

---

## Future Work

* Add **TreeQN** for temporal reasoning
* Add **MCTS-guided DQN** for planning-enhanced training

---

## License

This project is licensed under the [LICENSE](LICENSE) file.
