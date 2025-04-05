from particle_filter import TargetPF
from world_objects import Object, Location, Velocity
from utils import calculate_distance

class World():
    def __init__(self):
        self.estimated_landmark = TargetPF()
        true_landmark_location = Location(42, 42, 42)
        self.true_landmark = Object(location=true_landmark_location, name="true_landmark")
        agent_location = Location(0, 0, 0)
        self.agent = Object(location=agent_location, name="agent")
        self.objects = [self.estimated_landmark, self.true_landmark, self.agent]
        self.previous_range = calculate_distance(self.agent, self.true_landmark)
        self.reward = 0
    
    def step(self, action: Velocity):
        self.agent.velocity = action
        self.agent.update_position()
        self.true_landmark.update_position()
        self.previous_range = calculate_distance(self.agent, self.true_landmark)
        self.estimated_landmark.update_particle_filter(new_range=True, 
                                                       measurement=self.previous_range,
                                                       observer=self.agent.location)
        self.reward = 1 / (calculate_distance(self.estimated_landmark, self.true_landmark) + 1e-6)
        
    def encode_state(self):
        """Return the state as a tuple (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, previous_range)
         where landmark is the estimated landmark."""
        return (self.agent.location.x, self.agent.location.y, self.agent.velocity.x, self.agent.velocity.y,
                self.estimated_landmark.location.x, self.estimated_landmark.location.y, self.estimated_landmark.location.depth,
                self.previous_range)
    
    def decode_state(self, state: tuple):
        """Decode the state tuple into the agent and landmark locations and velocities."""
        agent_location = Location(state[0], state[1], 0)
        agent_velocity = Velocity(state[2], state[3], 0)
        landmark_location = Location(state[4], state[5], state[6])
        
        self.agent.location = agent_location
        self.agent.velocity = agent_velocity
        self.estimated_landmark.location = landmark_location
        
    def __str__(self):
        return f"""World:\n
        {"-"*15}\n
        reward: {self.reward},\n
        {"-"*15}\n
        agent: {self.agent},\n
        {"-"*15}\n
        true landmark: {self.true_landmark}\n, 
        {"-"*15} \n
        estimated landmark: {self.estimated_landmark}\n"""

if __name__ == "__main__":
    world = World()
    print(world)
    for i in range(10):
        world.step(Velocity(1, 1, 0))
    print(world)
    