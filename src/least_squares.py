# In least_squares.py

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import warnings
from collections import deque # <--- Import deque

from world_objects import Location, Velocity
from configs import LeastSquaresConfig

class TrackedTargetLS:

    def __init__(self, config: LeastSquaresConfig):
        self.config = config
        self.history_size = config.history_size
        self.min_points_required = max(config.min_points_required, 3)
        self.position_buffer_size = config.position_buffer_size
        self.velocity_smoothing = max(config.velocity_smoothing, 1)
        self.min_observer_movement = config.min_observer_movement
        self.location_smoothing_factor = config.location_smoothing_factor

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        self._is_initialized = False
        # --- Use deque for history ---
        self._observer_locations: deque[Location] = deque(maxlen=self.history_size)
        self._range_measurements: deque[float] = deque(maxlen=self.history_size)
        self._timestamps_of_measurements: deque[float] = deque(maxlen=self.history_size)
        self._position_history: deque[Tuple[float, Location]] = deque(maxlen=self.position_buffer_size)
        # --- End deque usage ---
        self._current_timestamp = 0.0

    def update(self,
              dt: float,
              has_new_range: bool,
              range_measurement: float,
              observer_location: Location,
              perform_update_step: bool = True):

        self._current_timestamp += dt
        new_measurement_added = False # Flag to track if LS might have new data

        if has_new_range and range_measurement > 0 and perform_update_step:
            # Use [-1] to access the last element of deque efficiently
            has_moved_enough = (not self._observer_locations or
                                self._calculate_distance(observer_location, self._observer_locations[-1]) > self.min_observer_movement)

            if has_moved_enough:
                observer_2d_loc = Location(x=observer_location.x, y=observer_location.y, depth=0.0)
                # Deque automatically handles maxlen - just append
                self._observer_locations.append(observer_2d_loc)
                self._range_measurements.append(range_measurement)
                self._timestamps_of_measurements.append(self._current_timestamp)
                new_measurement_added = True # Mark that we added data
                # No need for explicit pop(0) anymore

        # --- Prediction Step (Always apply if initialized) ---
        if self.estimated_location and self.estimated_velocity:
            predicted_location = Location(
                x=self.estimated_location.x + self.estimated_velocity.x * dt,
                y=self.estimated_location.y + self.estimated_velocity.y * dt,
                depth=self.estimated_location.depth
            )
            self.estimated_location = predicted_location
        # --- End Prediction Step ---

        # --- Least Squares Calculation and Soft Update ---
        raw_ls_estimate_xy = None
        if perform_update_step and new_measurement_added and len(self._observer_locations) >= self.min_points_required:
            raw_ls_estimate_xy = self._solve_least_squares_linearized()

        if raw_ls_estimate_xy is not None:
            raw_ls_location = Location(x=raw_ls_estimate_xy[0], y=raw_ls_estimate_xy[1], depth=0.0)

            if not self._is_initialized or self.estimated_location is None:
                self.estimated_location = raw_ls_location
                self._is_initialized = True
                # print(f"LS Initialized at t={self._current_timestamp:.2f}") # Keep for debug if needed
            else:
                alpha = self.location_smoothing_factor
                smooth_x = alpha * raw_ls_location.x + (1 - alpha) * self.estimated_location.x
                smooth_y = alpha * raw_ls_location.y + (1 - alpha) * self.estimated_location.y
                smoothed_location = Location(x=smooth_x, y=smooth_y, depth=self.estimated_location.depth)
                self.estimated_location = smoothed_location

            # Append smoothed estimate to deque (maxlen handled automatically)
            self._position_history.append((self._current_timestamp, self.estimated_location))
            self._update_velocity_estimate() # Update velocity based on smoothed history

    def _solve_least_squares_linearized(self) -> Optional[np.ndarray]:
        n_points = len(self._observer_locations)
        if n_points < 3:
            return None

        # Access first elements efficiently with [0] index
        ref_loc = self._observer_locations[0]
        ref_range_sq = self._range_measurements[0]**2
        x1, y1 = ref_loc.x, ref_loc.y

        A = np.zeros((n_points - 1, 2))
        b = np.zeros(n_points - 1)

        # Iterate through deque elements
        for i in range(1, n_points):
            obs_loc = self._observer_locations[i]
            range_sq = self._range_measurements[i]**2
            xi, yi = obs_loc.x, obs_loc.y

            A[i-1, 0] = 2 * (x1 - xi)
            A[i-1, 1] = 2 * (y1 - yi)
            b[i-1] = (range_sq - ref_range_sq) - (xi**2 - x1**2) - (yi**2 - y1**2)

        try:
            A_pinv = np.linalg.pinv(A)
            solution = A_pinv @ b
            return solution
        except np.linalg.LinAlgError:
            return None
        except Exception as e:
            warnings.warn(f"An unexpected error occurred in least squares: {e}", RuntimeWarning)
            return None

    def _update_velocity_estimate(self):
        if len(self._position_history) < 2:
            if self.estimated_velocity is None:
                self.estimated_velocity = Velocity(x=0.0, y=0.0, z=0.0)
            return

        num_diffs = min(self.velocity_smoothing, len(self._position_history) - 1)
        if num_diffs < 1: num_diffs = 1

        # Use negative indexing for deque (efficient access from right)
        older_idx = -num_diffs - 1
        newer_idx = -1

        # Check if older_idx is valid (deque handles negative indices differently than lists if too large)
        # It raises IndexError if index is out of bounds
        try:
            older_timestamp, older_pos = self._position_history[older_idx]
            newer_timestamp, newer_pos = self._position_history[newer_idx]
        except IndexError:
             # Fallback to using the oldest actual element if requested index is too far back
             older_timestamp, older_pos = self._position_history[0]
             newer_timestamp, newer_pos = self._position_history[-1]
             num_diffs = len(self._position_history) - 1 # Actual number of diffs used

        time_diff = newer_timestamp - older_timestamp

        if time_diff > 1e-6:
            vx = (newer_pos.x - older_pos.x) / time_diff
            vy = (newer_pos.y - older_pos.y) / time_diff
            self.estimated_velocity = Velocity(x=vx, y=vy, z=0.0)
        else:
            if not self.estimated_velocity:
                self.estimated_velocity = Velocity(x=0.0, y=0.0, z=0.0)

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)**0.5

    def encode_state(self) -> Dict[str, Any]:
        """ Encodes the internal state, converting deques to lists for saving. """
        state = {
            "config_dump": self.config.model_dump(),
            "is_initialized": self._is_initialized,
            "current_timestamp": self._current_timestamp,
            # Convert deques to lists for serialization
            "observer_locations": [(loc.x, loc.y, loc.depth) for loc in list(self._observer_locations)],
            "range_measurements": list(self._range_measurements),
            "position_history": [(ts, (pos.x, pos.y, pos.depth)) for ts, pos in list(self._position_history)],
            "timestamps_of_measurements": list(self._timestamps_of_measurements),
        }
        # ... (rest of encode_state is the same) ...
        if self.estimated_location is not None:
            loc = self.estimated_location
            state["estimated_location"] = (loc.x, loc.y, loc.depth)
        if self.estimated_velocity is not None:
            vel = self.estimated_velocity
            state["estimated_velocity"] = (vel.x, vel.y, vel.z)
        return state

    def decode_state(self, state_dict: Dict[str, Any]) -> None:
        """ Restores the internal state, creating deques from saved lists. """
        self.config = LeastSquaresConfig(**state_dict.get("config_dump", {}))
        self.location_smoothing_factor = self.config.location_smoothing_factor
        self.history_size = self.config.history_size # Ensure maxlen is set correctly
        self.position_buffer_size = self.config.position_buffer_size

        self._is_initialized = state_dict.get("is_initialized", False)
        self._current_timestamp = state_dict.get("current_timestamp", 0.0)

        # Recreate deques from lists, respecting maxlen
        self._observer_locations = deque(
            (Location(x=x, y=y, depth=z) for x, y, z in state_dict.get("observer_locations", [])),
            maxlen=self.history_size
        )
        self._range_measurements = deque(state_dict.get("range_measurements", []), maxlen=self.history_size)
        self._timestamps_of_measurements = deque(state_dict.get("timestamps_of_measurements", []), maxlen=self.history_size)

        pos_history_list = []
        for ts, (x, y, depth) in state_dict.get("position_history", []):
             pos_history_list.append((ts, Location(x=x, y=y, depth=depth)))
        self._position_history = deque(pos_history_list, maxlen=self.position_buffer_size)

        # ... (rest of decode_state is the same) ...
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