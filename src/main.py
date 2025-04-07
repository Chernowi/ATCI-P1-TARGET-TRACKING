from SAC import train_sac, evaluate_sac
from world import World
from configs import vast_config as config
import torch

if __name__ == "__main__":
    # Create world using config
    world = World(
        dt=config.world.dt,
        success_threshold=config.world.success_threshold
    )
    
    # Check for available GPUs
    use_multi_gpu = torch.cuda.device_count() > 1
    
    # Train SAC with GPU support
    print("Training SAC agent...")
    agent, rewards = train_sac(
        world, 
        num_episodes=config.training.num_episodes,
        max_steps=config.training.max_steps, 
        batch_size=config.training.batch_size,
        replay_buffer_size=config.training.replay_buffer_size,
        save_interval=config.training.save_interval,
        models_dir=config.training.models_dir,
        success_threshold=config.training.success_threshold,
        use_multi_gpu=use_multi_gpu  # Enable multi-GPU if available
    )
    
    # Save final model
    agent.save_model(f"{config.training.models_dir}/sac_final.pt")
    
    # Evaluate
    print("\nEvaluating SAC agent...")
    evaluate_sac(
        agent, 
        world, 
        num_episodes=config.evaluation.num_episodes,
        max_steps=config.evaluation.max_steps,
        render=config.evaluation.render,
        success_threshold=config.evaluation.success_threshold
    )
    
    print("\nTraining complete. Find output in the world_snapshots directory.")
    print("You can convert the frames to a video using ffmpeg.")