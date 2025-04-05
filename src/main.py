from sac import train_sac, evaluate_sac
from world import World

if __name__ == "__main__":
    # Example usage
    world = World(dt=1.0, success_threshold=2.0)
    
    # Train SAC
    print("Training SAC agent...")
    agent, rewards = train_sac(world, num_episodes=100, max_steps=500, success_threshold=2.0)
    
    # Save final model
    agent.save_model("sac_models/sac_final.pt")
    
    # Evaluate
    print("\nEvaluating SAC agent...")
    evaluate_sac(agent, world, num_episodes=1, max_steps=100, render=True, success_threshold=2.0)
    
    print("\nTraining complete. Find output in the world_snapshots directory.")
    print("You can convert the frames to a video using ffmpeg.")