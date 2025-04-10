# --- START OF FILE run_tsac_experiment.py ---

import os
import argparse
import torch

from TSAC import TSAC, evaluate_tsac # Import T-SAC specific classes/functions
from configs import CONFIGS, DefaultConfig
from visualization import reset_trajectories


def run_experiment(config_name: str, model_path: str, num_episodes: int, max_steps: int, render: bool):
    """
    Load a trained T-SAC model and run an experiment using specified configuration.

    Args:
        config_name: Name of the configuration profile to use (e.g., "tsac_default").
        model_path: Path to the saved T-SAC model checkpoint.
        num_episodes: Number of episodes to run for evaluation.
        max_steps: Maximum steps per evaluation episode.
        render: Whether to render the environment and save GIFs.
    """
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for T-SAC experiment")

    # Override specific evaluation parameters from command line args if provided explicitly
    if num_episodes is not None:
        config.evaluation.num_episodes = num_episodes
    if max_steps is not None:
        config.evaluation.max_steps = max_steps
    if render is not None:
        config.evaluation.render = render

    # Get configured device
    device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the T-SAC agent using the T-SAC config from the loaded DefaultConfig
    agent = TSAC(config=config.tsac, device=device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading T-SAC model from {model_path}...")

    agent.load_model(model_path)
    reset_trajectories()

    print(f"\nRunning experiment with T-SAC model {os.path.basename(model_path)}...")
    evaluate_tsac(agent=agent, config=config) # Use evaluate_tsac function

    print(
        f"\nT-SAC Experiment complete. Visualizations saved to {config.visualization.save_dir} directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an experiment with a trained T-SAC model")
    parser.add_argument(
        "--config", "-c", type=str, default="tsac_default", # Default to a T-SAC config
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="sac_models/tsac_final.pt", # Default model name
        help="Path to the trained T-SAC model checkpoint"
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=None,
        help="Number of episodes to run (overrides config)"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=None,
        help="Maximum steps per episode (overrides config)"
    )
    parser.add_argument(
        "--render", dest="render", action="store_true", default=None,
        help="Enable visualization rendering (overrides config)"
    )
    parser.add_argument(
        "--no-render", dest="render", action="store_false", default=None,
        help="Disable visualization rendering (overrides config)"
    )

    args = parser.parse_args()

    # Check if default model path used and directory needs creation
    default_model_dir = os.path.dirname("sac_models/tsac_final.pt")
    if args.model == "sac_models/tsac_final.pt" and not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
        print(f"Created directory: {default_model_dir}")
        print("Warning: Default model path specified, but directory didn't exist. Ensure tsac_final.pt is present or specify a valid model path.")

    run_experiment(
        config_name=args.config,
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render
    )