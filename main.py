import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run/train RL algorithms")
    parser.add_argument(
        "--train",
        type=str,
        choices=["qlearning", "dqn", "ppo"],
        help="Train the specified algorithm"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation script"
    )
    args = parser.parse_args()

    if args.train:
        script_map = {
            "qlearning": "scripts.train_qlearning",
            "dqn": "scripts.train_dqn",
            "ppo": "scripts.train_ppo",
        }
        module_name = script_map[args.train]
        subprocess.run(["python3", "-m", module_name], check=True)

    elif args.eval:
        subprocess.run(["python3", "-m", "scripts.evaluate_algo"], check=True)

    else:
        print("Please specify either --train <algo> or --eval")
