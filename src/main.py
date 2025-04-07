import argparse
import os
import torch

from SAC import train_sac, evaluate_sac
from world import World
from configs import CONFIGS, DefaultConfig


def main(config_name: str):
    """Main function to train and evaluate the SAC agent."""
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}'")

    use_multi_gpu = torch.cuda.device_count() > 1

    os.makedirs(config.training.models_dir, exist_ok=True)

    print("Training SAC agent...")
    agent, rewards = train_sac(config=config, use_multi_gpu=use_multi_gpu)

    final_model_path = os.path.join(config.training.models_dir, "sac_final.pt")
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    print("\nEvaluating SAC agent...")
    evaluate_sac(agent=agent, config=config)

    print(
        f"\nTraining and evaluation complete. Find output in the {config.visualization.save_dir} directory.")
    print("You can convert the frames to a video using ffmpeg or view the generated GIFs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate SAC agent for landmark tracking.")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    args = parser.parse_args()
    main(config_name=args.config)
