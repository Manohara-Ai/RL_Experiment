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

Example:

* `results/plots/DQN.png` → DQN learning curve
* `results/csv/Q_Learning.csv` → Q-Learning training log

---

## Future Work

* Add **TreeQN** for temporal reasoning
* Add **MCTS-guided DQN** for planning-enhanced training

---

## License

This project is licensed under the [LICENSE](LICENSE) file.
