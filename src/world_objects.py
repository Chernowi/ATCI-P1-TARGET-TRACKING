class Velocity():
    """Represents the velocity of an object in 3D space."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def is_moving(self) -> bool:
        """Check if any velocity component is non-zero."""
        return self.x != 0 or self.y != 0 or self.z != 0

    def __str__(self) -> str:
        """String representation of velocity."""
        return f"Vel:(vx:{self.x:.2f}, vy:{self.y:.2f}, vz:{self.z:.2f})"


class Location():
    """Represents the location of an object in 3D space (using 'depth' for z-axis)."""

    def __init__(self, x: float, y: float, depth: float):
        self.x = x
        self.y = y
        self.depth = depth

    def update(self, velocity: Velocity, dt: float = 1.0):
        """Update location based on velocity and time step."""
        self.x += velocity.x * dt
        self.y += velocity.y * dt
        self.depth += velocity.z * dt

    def __str__(self) -> str:
        """String representation of location."""
        return f"Pos:(x:{self.x:.2f}, y:{self.y:.2f}, d:{self.depth:.2f})"


class Object():
    """Represents a generic object with location and velocity."""

    def __init__(self, location: Location, velocity: Velocity = None, name: str = None):
        self.name = name if name else "Unnamed Object"
        self.location = location
        self.velocity = velocity if velocity is not None else Velocity(
            0.0, 0.0, 0.0)

    def update_position(self, dt: float = 1.0):
        """Update the object's position based on its velocity and time step."""
        if self.velocity and self.velocity.is_moving():
            self.location.update(self.velocity, dt)

    def __str__(self) -> str:
        """String representation of the object."""
        name_str = f"{self.name}: "
        return f"{name_str}{self.location}, {self.velocity}"
