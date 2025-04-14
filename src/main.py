import argparse
import os
import torch

from SAC import train_sac

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
        print(f"Overriding CUDA device: {cuda_device}")

    # Override algorithm if specified
    effective_algorithm = algorithm if algorithm else config.algorithm
    if algorithm and algorithm != config.algorithm:
         print(f"Overriding config algorithm '{config.algorithm}' with command line argument: '{algorithm}'")
         config.algorithm = algorithm # Ensure config object reflects the override
    elif not algorithm and config_name == "default":
         # If using default config and no algorithm specified, maybe pick one explicitly?
         print(f"Using algorithm specified in default config: '{config.algorithm}'")
    else:
         print(f"Using algorithm: '{effective_algorithm}'")


    use_multi_gpu = torch.cuda.device_count() > 1

    # Ensure model directory exists
    model_dir = config.training.models_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved in: {os.path.abspath(model_dir)}")

    # Select algorithm based on the effective choice
    # if effective_algorithm.lower() == "ppo":
    #     print("Training PPO agent...")
    #     agent, _ = train_ppo(config=config, use_multi_gpu=use_multi_gpu)
    #     final_model_path = os.path.join(model_dir, "ppo_final.pt")
    #     agent.save_model(final_model_path)
    #     print(f"Final PPO model saved to {final_model_path}")
    #     print("\nEvaluating PPO agent...")
    #     # evaluate_ppo(agent=agent, config=config)

    # elif effective_algorithm.lower() == "tsac": # Add T-SAC case
    #     print("Training T-SAC agent...")
    #     agent, _ = train_tsac(config=config, use_multi_gpu=use_multi_gpu)
    #     final_model_path = os.path.join(model_dir, "tsac_final.pt") # Save as tsac_final
    #     agent.save_model(final_model_path)
    #     print(f"Final T-SAC model saved to {final_model_path}")
    #     print("\nEvaluating T-SAC agent...")
    #     # evaluate_tsac(agent=agent, config=config) # Use evaluate_tsac

    # elif effective_algorithm.lower() == "sac": # Default/fallback to SAC
    print("Training SAC agent...")
    agent, _ = train_sac(config=config, use_multi_gpu=use_multi_gpu)
    final_model_path = os.path.join(model_dir, "sac_final.pt")
    agent.save_model(final_model_path)
    print(f"Final SAC model saved to {final_model_path}")
    print("\nEvaluating SAC agent...")
        # evaluate_sac(agent=agent, config=config)
    # else:
    #     raise ValueError(f"Unknown algorithm specified: {effective_algorithm}. Choose 'sac', 'ppo', or 'tsac'.")


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
        choices=["sac", "ppo", "tsac"], # Add tsac choice
        help="RL algorithm to use ('sac', 'ppo', 'tsac'). Overrides config."
    )
    args = parser.parse_args()
    main(config_name=args.config, cuda_device=args.device, algorithm=args.algorithm)