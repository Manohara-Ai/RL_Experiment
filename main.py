import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single RL algorithm")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["qlearning", "dqn"],
        help="Algorithm to run"
    )
    args = parser.parse_args()

    script_map = {
        "qlearning": "scripts.train_qlearning",
        "dqn": "scripts.train_dqn"
    }

    module_name = script_map[args.algo]

    subprocess.run(["python3", "-m", module_name], check=True)
