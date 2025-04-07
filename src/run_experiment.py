import os
import argparse
from SAC import SAC, evaluate_sac
from world import World
from configs import default_config
from visualization import reset_trajectories

def run_experiment(model_path, num_episodes=5, max_steps=100, render=True):
    """
    Load a trained SAC model and run an experiment to visualize its performance.
    
    Args:
        model_path: Path to the saved SAC model
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    """
    # Create world using config
    world = World(
        dt=default_config.world.dt,
        success_threshold=default_config.world.success_threshold
    )
    
    # Initialize SAC agent
    print(f"Loading model from {model_path}...")
    agent = SAC(
        state_dim=default_config.sac.state_dim,
        action_dim=default_config.sac.action_dim,
        action_scale=default_config.sac.action_scale,
        lr=default_config.sac.lr,
        gamma=default_config.sac.gamma,
        tau=default_config.sac.tau,
        alpha=default_config.sac.alpha,
        auto_tune_alpha=default_config.sac.auto_tune_alpha
    )
    agent.load_model(model_path)
    
    # Reset visualization trajectories
    reset_trajectories()
    
    # Evaluate the loaded model
    print(f"\nRunning experiment with model {os.path.basename(model_path)}...")
    evaluate_sac(
        agent,
        world,
        num_episodes=num_episodes,
        max_steps=max_steps,
        render=render,
        success_threshold=default_config.evaluation.success_threshold
    )
    
    print(f"\nExperiment complete. Visualizations saved to {default_config.visualization.save_dir} directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with a trained SAC model")
    parser.add_argument("--model", "-m", type=str, default="sac_models/sac_final.pt", 
                        help="Path to the trained SAC model")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--steps", "-s", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--no-render", action="store_false", dest="render",
                        help="Disable visualization rendering")
    args = parser.parse_args()
    
    run_experiment(args.model, args.episodes, args.steps, args.render)