from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import warnings

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

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        self._is_initialized = False
        self._observer_locations: List[Location] = []
        self._range_measurements: List[float] = []
        self._position_history: List[Tuple[float, Location]] = []
        self._timestamps_of_measurements: List[float] = []
        self._current_timestamp = 0.0

    def update(self,
              dt: float,
              has_new_range: bool,
              range_measurement: float,
              observer_location: Location,
              perform_update_step: bool = True):

        self._current_timestamp += dt

        if has_new_range and range_measurement > 0 and perform_update_step:
            has_moved_enough = (not self._observer_locations or
                                self._calculate_distance(observer_location, self._observer_locations[-1]) > self.min_observer_movement)

            if has_moved_enough:
                observer_2d_loc = Location(x=observer_location.x, y=observer_location.y, depth=0.0)
                self._observer_locations.append(observer_2d_loc)
                self._range_measurements.append(range_measurement)
                self._timestamps_of_measurements.append(self._current_timestamp)
                new_measurement_added = True

                if len(self._observer_locations) > self.history_size:
                    self._observer_locations.pop(0)
                    self._range_measurements.pop(0)
                    self._timestamps_of_measurements.pop(0)

        if not perform_update_step:
            if self.estimated_location and self.estimated_velocity:
                self.estimated_location.x += self.estimated_velocity.x * dt
                self.estimated_location.y += self.estimated_velocity.y * dt
            return

        position_estimate_xy = None
        if len(self._observer_locations) >= self.min_points_required:
            position_estimate_xy = self._solve_least_squares_linearized()

        if position_estimate_xy is not None:
            new_location = Location(x=position_estimate_xy[0], y=position_estimate_xy[1], depth=0.0)
            self._position_history.append((self._current_timestamp, new_location))
            if len(self._position_history) > self.position_buffer_size:
                self._position_history.pop(0)

            self.estimated_location = new_location
            self._is_initialized = True
            self._update_velocity_estimate()

        elif self._is_initialized and self.estimated_location and self.estimated_velocity:
            self.estimated_location.x += self.estimated_velocity.x * dt
            self.estimated_location.y += self.estimated_velocity.y * dt


    def _solve_least_squares_linearized(self) -> Optional[np.ndarray]:
        n_points = len(self._observer_locations)
        if n_points < 3:
            return None

        ref_loc = self._observer_locations[0]
        ref_range_sq = self._range_measurements[0]**2
        x1, y1 = ref_loc.x, ref_loc.y

        A = np.zeros((n_points - 1, 2))
        b = np.zeros(n_points - 1)

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
            warnings.warn("Least squares solution failed due to numerical instability.", RuntimeWarning)
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

        older_timestamp, older_pos = self._position_history[-num_diffs - 1]
        newer_timestamp, newer_pos = self._position_history[-1]

        time_diff = newer_timestamp - older_timestamp

        if time_diff > 1e-6:
            vx = (newer_pos.x - older_pos.x) / time_diff
            vy = (newer_pos.y - older_pos.y) / time_diff
            self.estimated_velocity = Velocity(x=vx, y=vy, z=0.0)
        else:
            if not self.estimated_velocity:
                self.estimated_velocity = Velocity(x=0.0, y=0.0, z=0.0)
            warnings.warn("Time difference for velocity calculation is near zero.", RuntimeWarning)

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)**0.5

    def encode_state(self) -> Dict[str, Any]:
        state = {
            "config": self.config.model_dump(),
            "is_initialized": self._is_initialized,
            "current_timestamp": self._current_timestamp,
            "observer_locations": [(loc.x, loc.y, loc.depth) for loc in self._observer_locations],
            "range_measurements": self._range_measurements.copy(),
            "position_history": [(ts, (pos.x, pos.y, pos.depth)) for ts, pos in self._position_history],
            "timestamps_of_measurements": self._timestamps_of_measurements.copy(),
        }

        if self.estimated_location is not None:
            state["estimated_location"] = (self.estimated_location.x, self.estimated_location.y, self.estimated_location.depth)

        if self.estimated_velocity is not None:
            state["estimated_velocity"] = (self.estimated_velocity.x, self.estimated_velocity.y, self.estimated_velocity.z)

        return state

    def decode_state(self, state_dict: Dict[str, Any]) -> None:
        self._is_initialized = state_dict.get("is_initialized", False)
        self._current_timestamp = state_dict.get("current_timestamp", 0.0)

        self._observer_locations = [Location(x=x, y=y, depth=z)
                                   for x, y, z in state_dict.get("observer_locations", [])]
        self._range_measurements = state_dict.get("range_measurements", [])
        self._timestamps_of_measurements = state_dict.get("timestamps_of_measurements", [])

        self._position_history = []
        for ts, (x, y, depth) in state_dict.get("position_history", []):
             self._position_history.append((ts, Location(x=x, y=y, depth=depth)))

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
