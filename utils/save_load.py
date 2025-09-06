import os
import numpy as np
import pandas as pd
import torch
import yaml

CSV_DIR = "results/csv"
CONFIG_DIR = "config"
CHECKPOINT_DIR = "results/checkpoints"

def save_csv(df, filename):
    path = os.path.join(CSV_DIR, filename)
    df.to_csv(path, index=False)
    print("CSV saved at:", os.path.abspath(path))

def load_csv(filename):
    path = os.path.join(CSV_DIR, filename)
    return pd.read_csv(path)

def load_config(filename):
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_checkpoint(model, optimizer, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

    print("Checkpoint saved at:", os.path.abspath(path))

def load_checkpoint(model, optimizer, filename, device="cpu"):
    path = os.path.join(CHECKPOINT_DIR, filename)
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer

def save_qtable(qtable, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    np.save(path, qtable)
    print("Q-table saved at:", os.path.abspath(path))

def load_qtable(filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    qtable = np.load(path)
    return qtable
