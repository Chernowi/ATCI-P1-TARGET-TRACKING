from SAC import train_sac, evaluate_sac
from world import World
from configs import default_config

if __name__ == "__main__":
    # Create world using config
    world = World(
        dt=default_config.world.dt,
        success_threshold=default_config.world.success_threshold
    )
    
    # Train SAC
    print("Training SAC agent...")
    agent, rewards = train_sac(
        world, 
        num_episodes=default_config.training.num_episodes,
        max_steps=default_config.training.max_steps, 
        batch_size=default_config.training.batch_size,
        replay_buffer_size=default_config.training.replay_buffer_size,
        save_interval=default_config.training.save_interval,
        models_dir=default_config.training.models_dir,
        success_threshold=default_config.training.success_threshold
    )
    
    # Save final model
    agent.save_model(f"{default_config.training.models_dir}/sac_final.pt")
    
    # Evaluate
    print("\nEvaluating SAC agent...")
    evaluate_sac(
        agent, 
        world, 
        num_episodes=default_config.evaluation.num_episodes,
        max_steps=default_config.evaluation.max_steps,
        render=default_config.evaluation.render,
        success_threshold=default_config.evaluation.success_threshold
    )
    
    print("\nTraining complete. Find output in the world_snapshots directory.")
    print("You can convert the frames to a video using ffmpeg.")