import os
import matplotlib.pyplot as plt

PLOT_DIR = "results/plots"

def plot_rewards_per_episode(res_all, title="agent", colors=("royalblue", "skyblue")):
    line_color, fill_color = colors
    plot_df = res_all.groupby("Episodes")["Rewards"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["Episodes"].to_numpy(),
             plot_df["mean"].to_numpy(),
             label="Q-Learning", c=line_color)
    
    plt.fill_between(plot_df["Episodes"].to_numpy(),
                     (plot_df["mean"] - plot_df["std"]).to_numpy(),
                     (plot_df["mean"] + plot_df["std"]).to_numpy(),
                     alpha=0.5, color=fill_color)

    plt.title(f"Reward per Episode - {title}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(PLOT_DIR, f"{title}.png")
    plt.savefig(save_path)
    print("Plot saved at:", os.path.abspath(save_path))
    plt.show()
