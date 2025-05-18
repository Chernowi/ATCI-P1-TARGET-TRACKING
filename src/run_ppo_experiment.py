import os
import argparse
import torch
import json

from PPO import PPO, evaluate_ppo 
from configs import CONFIGS, DefaultConfig, TrainingConfig, WorldConfig # Added WorldConfig

def run_experiment(config_name_arg: str, model_path_arg: str, num_episodes: int, max_steps: int, render: bool):
    experiment_dir_of_model = None
    actual_model_file_path = model_path_arg

    if os.path.isdir(model_path_arg):
        experiment_dir_of_model = model_path_arg
        models_subdir = os.path.join(experiment_dir_of_model, "models")
        if os.path.exists(models_subdir) and os.path.isdir(models_subdir):
            pt_files = [f for f in os.listdir(models_subdir) if f.endswith(".pt")]
            if not pt_files: raise FileNotFoundError(f"No .pt model files found in {models_subdir}")
            final_models = [f for f in pt_files if "final" in f.lower()]
            actual_model_file_path = os.path.join(models_subdir, sorted(final_models)[-1] if final_models else sorted(pt_files)[-1])
        else: raise FileNotFoundError(f"No 'models' subdirectory found in {experiment_dir_of_model}")
    elif os.path.isfile(model_path_arg):
        actual_model_file_path = model_path_arg
        experiment_dir_of_model = os.path.dirname(os.path.dirname(actual_model_file_path))
    else:
        temp_config = CONFIGS.get(config_name_arg, CONFIGS["default"])
        constructed_path = os.path.join(temp_config.training.models_dir, model_path_arg)
        if os.path.isfile(constructed_path):
            actual_model_file_path = constructed_path
            experiment_dir_of_model = os.path.dirname(os.path.dirname(actual_model_file_path))
            print(f"Interpreted model path relative to default models_dir: {actual_model_file_path}")
        else: raise FileNotFoundError(f"Model file or experiment directory not found: {model_path_arg}")

    if not os.path.exists(actual_model_file_path): raise FileNotFoundError(f"Model file not found: {actual_model_file_path}")
    print(f"Using model file: {os.path.abspath(actual_model_file_path)}")

    loaded_config_path = os.path.join(experiment_dir_of_model, "config.json")
    if os.path.exists(loaded_config_path):
        print(f"Loading configuration from: {loaded_config_path}")
        with open(loaded_config_path, 'r') as f: config_dict = json.load(f)
        config = DefaultConfig(**config_dict)
        config._experiment_path_for_vis = os.path.abspath(experiment_dir_of_model)
    else:
        print(f"Warning: config.json not found in {experiment_dir_of_model}. Using config '{config_name_arg}'.")
        if config_name_arg not in CONFIGS: raise ValueError(f"Unknown config: {config_name_arg}. Avail: {list(CONFIGS.keys())}")
        config = CONFIGS[config_name_arg].model_copy(deep=True)
        config._experiment_path_for_vis = os.getcwd() 
    print(f"Using effective algorithm: '{config.algorithm}' from loaded/selected config.")

    if num_episodes is not None: config.evaluation.num_episodes = num_episodes; print(f"Overriding num_episodes: {num_episodes}")
    if max_steps is not None: config.evaluation.max_steps = max_steps; print(f"Overriding max_steps: {max_steps}")
    if render is not None: config.evaluation.render = render; print(f"Setting render: {render}")

    device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # PPO agent needs PPOConfig, TrainingConfig, and WorldConfig
    agent = PPO(config=config.ppo, training_config=config.training, world_config=config.world, device=device)
    print(f"Loading PPO model weights from {os.path.abspath(actual_model_file_path)}...")
    agent.load_model(actual_model_file_path)

    print(f"\nRunning experiment with PPO model {os.path.basename(actual_model_file_path)}...")
    evaluate_ppo(agent=agent, config=config) 
    print(f"\nPPO Experiment complete.")
    if config.evaluation.render:
        vis_output_dir = config.visualization.save_dir
        if not os.path.isabs(vis_output_dir) and hasattr(config, '_experiment_path_for_vis'):
            vis_output_dir = os.path.join(config._experiment_path_for_vis, vis_output_dir)
        print(f"Visualizations potentially saved to {os.path.abspath(vis_output_dir)} directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with a trained PPO model")
    parser.add_argument("--config", "-c", type=str, default="default", help=f"Fallback config if not found. Avail: {list(CONFIGS.keys())}")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to PPO model (.pt) OR experiment directory.")
    parser.add_argument("--episodes", "-e", type=int, default=None, help="Num episodes (overrides config)")
    parser.add_argument("--steps", "-s", type=int, default=None, help="Max steps per episode (overrides config)")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", default=None, help="Enable rendering (overrides config)")
    render_group.add_argument("--no-render", dest="render", action="store_false", help="Disable rendering (overrides config)")
    args = parser.parse_args()
    run_experiment(args.config, args.model, args.episodes, args.steps, args.render)