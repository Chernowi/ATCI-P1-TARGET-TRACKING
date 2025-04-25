import os
import argparse
import torch

from SAC import SAC, evaluate_sac # evaluate_sac handles conditional visualization import
from configs import CONFIGS, DefaultConfig
# from visualization import reset_trajectories # Moved inside evaluate_sac

def run_experiment(config_name: str, model_path: str, num_episodes: int, max_steps: int, render: bool):
    """
    Load a trained SAC model and run an experiment using specified configuration.
    """
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for experiment")

    # Override evaluation parameters if provided
    if num_episodes is not None:
        config.evaluation.num_episodes = num_episodes
        print(f"Overriding num_episodes: {num_episodes}")
    if max_steps is not None:
        config.evaluation.max_steps = max_steps
        print(f"Overriding max_steps: {max_steps}")
    # Render is a boolean, check if it was explicitly set (not None)
    if render is not None:
        config.evaluation.render = render
        print(f"Setting render: {render}")

    device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use SAC config from the loaded DefaultConfig
    # SAC constructor requires sac_config, world_config, and device
    agent = SAC(config=config.sac, world_config=config.world, device=device)

    if not os.path.exists(model_path):
        # Try default directory if only filename is given
        default_dir_path = os.path.join(config.training.models_dir, model_path)
        if os.path.exists(default_dir_path):
             model_path = default_dir_path
        else:
             raise FileNotFoundError(f"Model file not found: {model_path} or {default_dir_path}")

    print(f"Loading SAC model from {model_path}...")
    agent.load_model(model_path)
    # reset_trajectories() # Moved to evaluate_sac

    print(f"\nRunning experiment with SAC model {os.path.basename(model_path)}...")
    # evaluate_sac now handles rendering and trajectory reset based on config
    evaluate_sac(agent=agent, config=config)

    print(f"\nExperiment complete.")
    if config.evaluation.render:
         print(f"Visualizations potentially saved to {config.visualization.save_dir} directory (if libraries were available).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with a trained SAC model")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="sac_final.pt", # Default filename
        help="Path to the trained SAC model checkpoint (relative to default models dir or absolute)"
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=None,
        help="Number of episodes to run (overrides config)"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=None,
        help="Maximum steps per episode (overrides config)"
    )
    # Mutually exclusive group for render/no-render
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

    # Construct full model path if only filename was given
    model_path = args.model
    config_to_get_dir = CONFIGS.get(args.config, CONFIGS["default"])
    models_dir = config_to_get_dir.training.models_dir
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = os.path.join(models_dir, model_path)

    # Ensure model dir exists if using default name/path
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
        render=args.render # Pass the potentially overridden value
    )