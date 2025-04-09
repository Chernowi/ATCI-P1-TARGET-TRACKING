from typing import Optional, List, Dict, Any
import numpy as np

from world_objects import Location, Velocity
from configs import LeastSquaresConfig

class TrackedTargetLS:
    """
    Represents a target whose state is estimated using a Least Squares approach.
    This class provides an alternative to the Particle Filter implementation.
    """

    def __init__(self, config: LeastSquaresConfig):
        """
        Initializes the target tracker with a Least Squares estimator.

        Args:
            config: Configuration object for the least squares estimator.
        """
        self.config = config
        self.history_size = config.history_size
        self.min_points_required = config.min_points_required
        self.position_buffer_size = config.position_buffer_size
        self.velocity_smoothing = config.velocity_smoothing
        self.min_observer_movement = config.min_observer_movement

        # Estimation results
        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None
        
        # Internal state
        self._is_initialized = False
        self._observer_locations: List[Location] = []
        self._range_measurements: List[float] = []
        self._position_history: List[Location] = []
        self._timestamp_history: List[float] = []
        self._current_timestamp = 0.0

    def update(self,
              dt: float,
              has_new_range: bool,
              range_measurement: float,
              observer_location: Location,
              perform_update_step: bool = True):
        """
        Updates the target estimate using least squares multilateration.

        Args:
            dt: Time step since the last update.
            has_new_range: Boolean indicating if range_measurement is new data.
            range_measurement: The measured range.
            observer_location: Current location of the observer/sensor.
            perform_update_step: If False, only prediction is done.
        """
        # Update timestamp
        self._current_timestamp += dt
        
        # Store new measurements
        if has_new_range and range_measurement > 0:
            # Only store measurements when the observer has moved enough
            if not self._observer_locations or self._calculate_distance(
                    observer_location, self._observer_locations[-1]) > self.min_observer_movement:
                self._observer_locations.append(Location(
                    x=observer_location.x, y=observer_location.y, depth=observer_location.depth))
                self._range_measurements.append(range_measurement)
                self._timestamp_history.append(self._current_timestamp)
                
                # Trim history if needed
                if len(self._observer_locations) > self.history_size:
                    self._observer_locations.pop(0)
                    self._range_measurements.pop(0)
                    self._timestamp_history.pop(0)
        
        # If we don't have enough points, can't estimate yet
        if len(self._observer_locations) < self.min_points_required:
            if self.estimated_location is None:
                # Initialize with a very rough guess
                last_observer = self._observer_locations[-1] if self._observer_locations else observer_location
                last_range = self._range_measurements[-1] if self._range_measurements else 10.0
                self.estimated_location = Location(
                    x=last_observer.x + last_range, 
                    y=last_observer.y, 
                    depth=0.0)
                self._is_initialized = False
            # Leave velocity at None or previous value
            return
        
        if not perform_update_step:
            # Only predict based on previous velocity
            if self.estimated_location and self.estimated_velocity:
                self.estimated_location.x += self.estimated_velocity.x * dt
                self.estimated_location.y += self.estimated_velocity.y * dt
            return
            
        # Perform least squares estimation
        position_estimate = self._solve_least_squares()
        
        if position_estimate is not None:
            # Create location object from estimate
            new_location = Location(
                x=position_estimate[0], 
                y=position_estimate[1], 
                depth=0.0)  # Depth is not estimated
                
            # Store in position history for velocity calculation
            self._position_history.append(new_location)
            if len(self._position_history) > self.position_buffer_size:
                self._position_history.pop(0)
                
            # Update estimated location
            self.estimated_location = new_location
            self._is_initialized = True
            
            # Calculate velocity if we have enough position history
            self._update_velocity_estimate()
    
    def _solve_least_squares(self) -> Optional[np.ndarray]:
        """
        Solves the multilateration problem using least squares.
        
        Returns:
            Numpy array with [x, y] position estimate, or None if estimation fails.
        """
        if len(self._observer_locations) < self.min_points_required:
            return None
            
        # Set up matrices for least squares
        n = len(self._observer_locations)
        A = np.zeros((n, 2))
        b = np.zeros(n)
        
        # Set up the system of equations
        for i in range(n):
            A[i, 0] = 2 * self._observer_locations[i].x
            A[i, 1] = 2 * self._observer_locations[i].y
            b[i] = (self._observer_locations[i].x**2 + 
                    self._observer_locations[i].y**2 - 
                    self._range_measurements[i]**2)
        
        # Solve the system using least squares
        try:
            position = np.linalg.lstsq(A, b, rcond=None)[0]
            return position
        except np.linalg.LinAlgError:
            print("WARNING: Least squares solution failed due to singular matrix.")
            return None
    
    def _update_velocity_estimate(self):
        """Updates the velocity estimate based on position history."""
        if len(self._position_history) < 2:
            self.estimated_velocity = Velocity(x=0.0, y=0.0, z=0.0)
            return
            
        # Calculate velocity based on position changes
        time_window = min(self.velocity_smoothing, len(self._position_history) - 1)
        older_pos = self._position_history[-time_window-1]
        newer_pos = self._position_history[-1]
        
        # Calculate time difference
        time_diff = 0.0
        if len(self._timestamp_history) >= time_window + 1:
            start_idx = len(self._timestamp_history) - time_window - 1
            time_diff = self._timestamp_history[-1] - self._timestamp_history[start_idx]
        
        if time_diff > 1e-6:
            vx = (newer_pos.x - older_pos.x) / time_diff
            vy = (newer_pos.y - older_pos.y) / time_diff
        else:
            # Fallback to previous velocity or zero
            if self.estimated_velocity:
                vx, vy = self.estimated_velocity.x, self.estimated_velocity.y
            else:
                vx, vy = 0.0, 0.0
                
        self.estimated_velocity = Velocity(x=vx, y=vy, z=0.0)
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Helper to calculate distance between two locations."""
        return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)**0.5

    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the internal state of the least squares estimator.
        
        Returns:
            Dictionary containing the serialized state
        """
        state = {
            "is_initialized": self._is_initialized,
            "current_timestamp": self._current_timestamp,
            "observer_locations": [(loc.x, loc.y, loc.depth) for loc in self._observer_locations],
            "range_measurements": self._range_measurements.copy(),
            "position_history": [(pos.x, pos.y, pos.depth) for pos in self._position_history],
            "timestamp_history": self._timestamp_history.copy(),
        }
        
        # Include estimated location and velocity if available
        if self.estimated_location is not None:
            state["estimated_location"] = (self.estimated_location.x, self.estimated_location.y, self.estimated_location.depth)
        
        if self.estimated_velocity is not None:
            state["estimated_velocity"] = (self.estimated_velocity.x, self.estimated_velocity.y, self.estimated_velocity.z)
            
        return state
    
    def decode_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Restores the internal state of the least squares estimator.
        
        Args:
            state_dict: Dictionary containing the serialized state
        """
        # Restore simple variables
        self._is_initialized = state_dict.get("is_initialized", False)
        self._current_timestamp = state_dict.get("current_timestamp", 0.0)
        
        # Restore collections
        self._observer_locations = [Location(x=x, y=y, depth=z) 
                                   for x, y, z in state_dict.get("observer_locations", [])]
        self._range_measurements = state_dict.get("range_measurements", [])
        self._position_history = [Location(x=x, y=y, depth=z) 
                                 for x, y, z in state_dict.get("position_history", [])]
        self._timestamp_history = state_dict.get("timestamp_history", [])
        
        # Restore estimated location and velocity
        if "estimated_location" in state_dict:
            x, y, depth = state_dict["estimated_location"]
            self.estimated_location = Location(x=x, y=y, depth=depth)
        else:
            self.estimated_location = None
            
        if "estimated_velocity" in state_dict:
            x, y, z = state_dict["estimated_velocity"]
            self.estimated_velocity = Velocity(x=x, y=y, z=z)
        else:
            self.estimated_velocity = None