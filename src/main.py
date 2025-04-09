import argparse
import os
import torch

from SAC import train_sac, evaluate_sac
from PPO import train_ppo, evaluate_ppo
from configs import CONFIGS, DefaultConfig


def main(config_name: str, cuda_device: str = None, algorithm: str = None):
    """Main function to train and evaluate the RL agent."""
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}'")
    
    # Override CUDA device if specified
    if cuda_device:
        config.cuda_device = cuda_device
        print(f"Using CUDA device: {cuda_device}")
    
    # Override algorithm if specified
    if algorithm:
        config.algorithm = algorithm
        print(f"Using algorithm: {algorithm}")
        
    use_multi_gpu = torch.cuda.device_count() > 1

    os.makedirs(config.training.models_dir, exist_ok=True)

    if config.algorithm.lower() == "ppo":
        print("Training PPO agent...")
        agent, _ = train_ppo(config=config, use_multi_gpu=use_multi_gpu)
        final_model_path = os.path.join(config.training.models_dir, "ppo_final.pt")
        agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        print("\nEvaluating PPO agent...")
        evaluate_ppo(agent=agent, config=config)
    else:  # Default to SAC
        print("Training SAC agent...")
        agent, _ = train_sac(config=config, use_multi_gpu=use_multi_gpu)
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
        description="Train and evaluate RL agent for landmark tracking.")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')"
    )
    parser.add_argument(
        "--algorithm", "-a", type=str, default=None,
        choices=["sac", "ppo"],
        help="RL algorithm to use ('sac' or 'ppo')"
    )
    args = parser.parse_args()
    main(config_name=args.config, cuda_device=args.device, algorithm=args.algorithm)
