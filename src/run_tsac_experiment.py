import os
import argparse
import torch

from TSAC import TSAC, evaluate_tsac # evaluate_tsac handles conditional visualization import
from configs import CONFIGS, DefaultConfig
# from visualization import reset_trajectories # Moved inside evaluate_tsac

def run_experiment(config_name: str, model_path: str, num_episodes: int, max_steps: int, render: bool):
    """
    Load a trained T-SAC model and run an experiment using specified configuration.
    """
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for T-SAC experiment")

    # Override evaluation parameters if provided
    if num_episodes is not None:
        config.evaluation.num_episodes = num_episodes
        print(f"Overriding num_episodes: {num_episodes}")
    if max_steps is not None:
        config.evaluation.max_steps = max_steps
        print(f"Overriding max_steps: {max_steps}")
    if render is not None:
        config.evaluation.render = render
        print(f"Setting render: {render}")

    device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use T-SAC config from the loaded DefaultConfig
    agent = TSAC(config=config.tsac, device=device)

    if not os.path.exists(model_path):
        default_dir_path = os.path.join(config.training.models_dir, model_path)
        if os.path.exists(default_dir_path):
             model_path = default_dir_path
        else:
             raise FileNotFoundError(f"Model file not found: {model_path} or {default_dir_path}")

    print(f"Loading T-SAC model from {model_path}...")
    agent.load_model(model_path)
    # reset_trajectories() # Moved to evaluate_tsac

    print(f"\nRunning experiment with T-SAC model {os.path.basename(model_path)}...")
    # evaluate_tsac handles rendering and trajectory reset based on config
    evaluate_tsac(agent=agent, config=config)

    print(f"\nT-SAC Experiment complete.")
    if config.evaluation.render:
         print(f"Visualizations potentially saved to {config.visualization.save_dir} directory (if libraries were available).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with a trained T-SAC model")
    parser.add_argument(
        "--config", "-c", type=str, default="tsac_default", # Default to a T-SAC config
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="tsac_final.pt", # Default T-SAC filename
        help="Path to the trained T-SAC model checkpoint (relative to default models dir or absolute)"
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=None,
        help="Number of episodes to run (overrides config)"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=None,
        help="Maximum steps per episode (overrides config)"
    )
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--render", action="store_true", default=None,
        help="Enable visualization rendering (overrides config's default)"
    )
    render_group.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Disable visualization rendering (overrides config's default)"
    )

    args = parser.parse_args()

    # Construct full model path if needed
    model_path = args.model
    config_to_get_dir = CONFIGS.get(args.config, CONFIGS["default"])
    models_dir = config_to_get_dir.training.models_dir
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = os.path.join(models_dir, model_path)

    if not os.path.exists(os.path.dirname(model_path)):
         try:
              os.makedirs(os.path.dirname(model_path))
              print(f"Created model directory: {os.path.dirname(model_path)}")
         except OSError as e:
              print(f"Warning: Could not create model directory {os.path.dirname(model_path)}: {e}")

    run_experiment(
        config_name=args.config,
        model_path=model_path,
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render
    )