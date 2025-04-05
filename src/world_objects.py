class Velocity():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def is_moving(self):
        return self.x != 0 or self.y != 0 or self.z != 0

    def __str__(self):
        return f"velocity: (vx:{self.x}, vy:{self.y}, vz:{self.z})"
    
class Location():
    def __init__(self, x: float, y: float, depth: float):
        self.x = x
        self.y = y
        self.depth = depth
        
    def update(self, velocity: Velocity):
        self.x += velocity.x
        self.y += velocity.y
        self.depth += velocity.z

class Object():
    def __init__(self, location: Location, velocity: Velocity = None, name: str = None):
        self.name = name
        self.location = location
        self.velocity = velocity
        
    def update_position(self):
        if self.velocity is not None and self.velocity.is_moving():
            self.location.update(self.velocity)
            
    def __str__(self):
        name_str = f"object: {self.name}, " if self.name else ""
        velocity_str = ", " + str(self.velocity) if self.velocity else ""
        return f"""{name_str} position: (x:{self.location.x},
                y:{self.location.y}, depth:{self.location.depth}) {velocity_str}"""