import os
import argparse
from SAC import SAC, evaluate_sac
from world import World
from configs import CONFIGS, DefaultConfig
from visualization import reset_trajectories

def run_experiment(config_name: str, model_path: str, num_episodes: int, max_steps: int, render: bool):
    """
    Load a trained SAC model and run an experiment using specified configuration.

    Args:
        config_name: Name of the configuration profile to use (e.g., "default").
        model_path: Path to the saved SAC model checkpoint.
        num_episodes: Number of episodes to run for evaluation.
        max_steps: Maximum steps per evaluation episode.
        render: Whether to render the environment and save GIFs.
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for experiment")

    # Override specific evaluation parameters from command line args if provided explicitly
    if num_episodes is not None:
        config.evaluation.num_episodes = num_episodes
    if max_steps is not None:
        config.evaluation.max_steps = max_steps
    if render is not None:
        config.evaluation.render = render

    agent = SAC(config=config.sac)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from {model_path}...")
    agent.load_model(model_path)

    reset_trajectories()

    print(f"\nRunning experiment with model {os.path.basename(model_path)}...")
    evaluate_sac(agent=agent, config=config)

    print(f"\nExperiment complete. Visualizations saved to {config.visualization.save_dir} directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with a trained SAC model")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="sac_models/sac_final.pt",
        help="Path to the trained SAC model checkpoint"
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
    if args.model == "sac_models/sac_final.pt" and not os.path.exists(os.path.dirname(args.model)):
         os.makedirs(os.path.dirname(args.model))
         print(f"Created directory: {os.path.dirname(args.model)}")
         print("Warning: Default model path specified, but directory didn't exist. Ensure sac_final.pt is present or specify a valid model path.")

    run_experiment(
        config_name=args.config,
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render
    )