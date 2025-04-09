"""
Particle Filter implementation for target tracking, using Location/Velocity objects.
Original Author: Ivan Masmitja Rusinol (March 29, 2020)
Refactored for clarity and integration with world_objects.
Project: AIforUTracking / Refactoring Exercise
"""
from typing import List, Optional, Dict, Any
import random
import numpy as np

from world_objects import Location, Velocity
from configs import ParticleFilterConfig

# --- Constants ---
SOUND_SPEED = 1500.0

# --- Particle Filter Core Logic ---


class ParticleFilterCore:
    """ Core implementation of the Particle Filter algorithm. """

    def __init__(self, config: ParticleFilterConfig):
        """
        Initializes the core particle filter components using configuration.

        Args:
            config: Configuration object for the particle filter.
        """
        self.state_dimension = 4  # [x, vx, y, vy]

        self.config = config

        self.num_particles = config.num_particles
        self.estimation_method = config.estimation_method
        self.max_particle_range = config.max_particle_range

        self.process_noise_position = config.process_noise_pos
        self.process_noise_orientation = config.process_noise_orient
        self.process_noise_velocity = config.process_noise_vel
        self.measurement_noise_stddev = config.measurement_noise_stddev

        self.particles_state = np.zeros(
            (self.num_particles, self.state_dimension))
        self.previous_particles_state = np.zeros(
            (self.num_particles, self.state_dimension))

        self.weights = np.ones(self.num_particles) / self.num_particles

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        self.position_covariance_matrix = np.eye(2)
        self.position_covariance_eigenvalues = np.array([0.02, 0.02])
        self.position_covariance_orientation = 0.0

        self.is_initialized = False
        self.previous_observer_location: Optional[Location] = None

    def initialize_particles(self, observer_location: Location, initial_range_guess: float):
        """
        Initializes particle positions and velocities around an initial guess.

        Args:
            observer_location: The initial location of the observer/sensor.
            initial_range_guess: An initial guess for the range to the target.
        """
        initial_range_stddev = self.config.initial_range_stddev
        initial_velocity_guess = self.config.initial_velocity_guess

        for i in range(self.num_particles):
            angle = random.uniform(0, 2 * np.pi)

            if self.estimation_method == 'area':
                radius = random.uniform(-self.max_particle_range,
                                        self.max_particle_range)
            else:
                radius = random.gauss(
                    initial_range_guess, initial_range_stddev)

            self.particles_state[i, 0] = observer_location.x + \
                radius * np.cos(angle)
            self.particles_state[i, 2] = observer_location.y + \
                radius * np.sin(angle)

            initial_orientation = random.uniform(0, 2 * np.pi)
            velocity_magnitude = abs(random.gauss(
                initial_velocity_guess, initial_velocity_guess / 2.0 + 1e-6))
            self.particles_state[i, 1] = velocity_magnitude * \
                np.cos(initial_orientation)
            self.particles_state[i, 3] = velocity_magnitude * \
                np.sin(initial_orientation)

        self.weights.fill(1.0 / self.num_particles)
        self.estimate_target_state()
        self.is_initialized = True

    def predict(self, dt: float):
        """
        Predicts the next state of each particle based on a simple motion model.

        Args:
            dt: Time step interval.
        """
        if not self.is_initialized:
            return

        use_gaussian_noise = False  # Keep original uniform noise behavior for process model

        for i in range(self.num_particles):
            vx = self.particles_state[i, 1]
            vy = self.particles_state[i, 3]
            current_orientation = np.arctan2(vy, vx)

            if use_gaussian_noise:
                orientation_noise = random.gauss(
                    0.0, self.process_noise_orientation)
            else:
                orientation_noise = random.uniform(
                    -self.process_noise_orientation, self.process_noise_orientation)

            new_orientation = (current_orientation +
                               orientation_noise) % (2 * np.pi)

            current_velocity_magnitude = np.sqrt(vx**2 + vy**2)

            if use_gaussian_noise:
                velocity_noise = random.gauss(0.0, self.process_noise_velocity)
            else:
                velocity_noise = random.uniform(
                    -self.process_noise_velocity, self.process_noise_velocity)

            new_velocity_magnitude = max(
                0.0, current_velocity_magnitude + velocity_noise)

            distance_travelled = current_velocity_magnitude * dt

            if use_gaussian_noise:
                position_noise = random.gauss(0.0, self.process_noise_position)
            else:
                position_noise = random.uniform(
                    -self.process_noise_position, self.process_noise_position)

            effective_distance = distance_travelled + position_noise

            self.particles_state[i, 0] += effective_distance * \
                np.cos(new_orientation)
            self.particles_state[i, 2] += effective_distance * \
                np.sin(new_orientation)
            self.particles_state[i, 1] = new_velocity_magnitude * \
                np.cos(new_orientation)
            self.particles_state[i, 3] = new_velocity_magnitude * \
                np.sin(new_orientation)

    def _calculate_likelihood(self,
                              predicted_range: float,
                              measured_range: float) -> float:
        """
        Calculates the likelihood weight based on the estimation method.

        Args:
            predicted_range: The range predicted by a particle.
            measured_range: The actual measured range (-1 if no measurement).

        Returns:
            The likelihood weight for the particle.
        """
        noise_stddev = self.measurement_noise_stddev

        if self.estimation_method == 'area':
            sigma_area = 1.0

            if measured_range != -1:
                return (0.5) - (1.0 / np.pi) * np.arctan((predicted_range - self.max_particle_range) / sigma_area)
            else:
                sigma_area = 40.0
                return (0.5) + (1.0 / np.pi) * np.arctan((predicted_range - self.max_particle_range) / sigma_area)

        elif self.estimation_method == 'range':
            if measured_range == -1:
                return 1.0

            variance = noise_stddev ** 2
            if variance < 1e-9:
                variance = 1e-9
            exponent = -((predicted_range - measured_range)
                         ** 2) / (2 * variance)
            denominator = np.sqrt(2 * np.pi * variance)
            likelihood = np.exp(exponent) / denominator
            return likelihood + 1e-9  # Add epsilon to prevent zero weights

        else:
            raise ValueError(
                f"Unknown estimation_method: {self.estimation_method}")

    def update_weights(self, measured_range: float, observer_location: Location):
        """
        Updates particle weights based on the latest measurement.

        Args:
            measured_range: The measured range to the target (-1 if no new measurement).
            observer_location: The location of the observer/sensor at the time of measurement.
        """
        if not self.is_initialized:
            return

        self.previous_observer_location = observer_location

        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        predicted_ranges = np.sqrt(dx**2 + dy**2)

        for i in range(self.num_particles):
            likelihood = self._calculate_likelihood(
                predicted_ranges[i], measured_range)
            self.weights[i] *= likelihood

        total_weight = np.sum(self.weights)
        if total_weight > 1e-9:
            self.weights /= total_weight
        else:
            print(
                "Warning: Particle weights sum near zero. Reinitializing weights uniformly.")
            self.weights.fill(1.0 / self.num_particles)

    def resample_particles(self, method: int):
        """
        Resamples particles based on their weights to combat particle degeneracy.

        Args:
            method: The resampling algorithm code (defined in config, passed by TrackedTargetPF).
        """
        if not self.is_initialized:
            return

        new_particles_state = np.zeros_like(self.particles_state)
        n = self.num_particles
        initial_velocity_guess = self.config.initial_velocity_guess

        if method == 1:  # Multinomial Resampling
            index = int(random.random() * n)
            beta = 0.0
            max_weight = np.max(self.weights)
            for i in range(n):
                beta += random.random() * 2.0 * max_weight
                while beta > self.weights[index]:
                    beta -= self.weights[index]
                    index = (index + 1) % n
                new_particles_state[i, :] = self.particles_state[index, :]

        elif method == 2:  # Systematic Resampling
            cumulative_sum = np.cumsum(self.weights)
            step = 1.0 / n
            u = random.uniform(0, step)
            i = 0
            for j in range(n):
                while u > cumulative_sum[i]:
                    i += 1
                new_particles_state[j, :] = self.particles_state[i, :]
                u += step

        elif method == 3 or method == 3.2:  # Hybrid: Systematic + Random Injection
            random_injection_ratio = 0.05  # Fixed ratio for random injection
            num_random = int(n * random_injection_ratio)
            num_systematic = n - num_random

            if num_systematic > 0:
                temp_particles_systematic = np.zeros(
                    (num_systematic, self.state_dimension))
                cumulative_sum = np.cumsum(self.weights)
                effective_total_weight = cumulative_sum[-1] if cumulative_sum[-1] > 1e-9 else 1.0
                step = effective_total_weight / num_systematic
                u = random.uniform(0, step)
                i = 0
                current_weight_sum = 0.0
                for j in range(num_systematic):
                    target_weight = u + j * step
                    while i < n - 1 and current_weight_sum + self.weights[i] < target_weight:
                        current_weight_sum += self.weights[i]
                        i += 1
                    temp_particles_systematic[j,
                                              :] = self.particles_state[i, :]
                new_particles_state[:num_systematic,
                                    :] = temp_particles_systematic

            center_x, center_y, injection_radius = None, None, None
            injection_possible = False
            if num_random > 0:
                if method == 3 and self.estimated_location is not None:
                    center_x = self.estimated_location.x
                    center_y = self.estimated_location.y
                    injection_radius = 0.2  # Fixed radius for estimate-centered injection
                    injection_possible = True
                elif method == 3.2 and self.previous_observer_location is not None:
                    center_x = self.previous_observer_location.x
                    center_y = self.previous_observer_location.y
                    injection_radius = self.max_particle_range  # Use configured range
                    injection_possible = True

                if not injection_possible:
                    print(
                        "Warning: Cannot perform random injection for resampling method", method)
                    if num_random > 0 and num_systematic > 0:
                        indices_to_copy = np.argsort(
                            self.weights)[-num_random:]
                        new_particles_state[num_systematic:,
                                            :] = self.particles_state[indices_to_copy, :]
                    elif num_random > 0:
                        new_particles_state[num_systematic:,
                                            :] = self.particles_state[num_systematic:, :]
                else:
                    for k in range(num_random):
                        idx = num_systematic + k
                        angle = random.uniform(0, 2 * np.pi)
                        radius = random.uniform(0, injection_radius)
                        new_particles_state[idx, 0] = center_x + \
                            radius * np.cos(angle)
                        new_particles_state[idx, 2] = center_y + \
                            radius * np.sin(angle)

                        orientation = random.uniform(0, 2 * np.pi)
                        velocity_magnitude = abs(random.gauss(
                            initial_velocity_guess, initial_velocity_guess / 2.0 + 1e-6))
                        new_particles_state[idx,
                                            1] = velocity_magnitude * np.cos(orientation)
                        new_particles_state[idx,
                                            3] = velocity_magnitude * np.sin(orientation)

        else:
            raise ValueError(f"Unsupported resampling method: {method}")

        self.particles_state = new_particles_state
        self.weights.fill(1.0 / self.num_particles)

    def estimate_target_state(self, method: int = 2):
        """
        Estimates the target state (location and velocity) from the particles.

        Args:
            method: 1 for simple mean, 2 for weighted mean (default).
        """
        if not self.is_initialized or self.num_particles == 0:
            self.estimated_location = None
            self.estimated_velocity = None
            return

        mean_state = np.zeros(self.state_dimension)
        total_weight = np.sum(self.weights)

        if method == 1:
            mean_state = np.mean(self.particles_state, axis=0)
        elif method == 2:
            if total_weight < 1e-9:
                mean_state = np.mean(self.particles_state, axis=0)
            else:
                mean_state = np.average(
                    self.particles_state, axis=0, weights=self.weights)
        else:
            raise ValueError(f"Unsupported estimation method: {method}")

        self.estimated_location = Location(
            x=mean_state[0], y=mean_state[2], depth=0.0)
        self.estimated_velocity = Velocity(
            x=mean_state[1], y=mean_state[3], z=0.0)

        try:
            if method == 2 and total_weight > 1e-9:
                x_coords = self.particles_state[:, 0]
                y_coords = self.particles_state[:, 2]
                avg_x = self.estimated_location.x
                avg_y = self.estimated_location.y
                cov_xx = np.sum(self.weights * (x_coords - avg_x)**2)
                cov_yy = np.sum(self.weights * (y_coords - avg_y)**2)
                cov_xy = np.sum(self.weights * (x_coords - avg_x)
                                * (y_coords - avg_y))
                self.position_covariance_matrix = np.array(
                    [[cov_xx, cov_xy], [cov_xy, cov_yy]])
            else:
                self.position_covariance_matrix = np.cov(
                    self.particles_state[:, 0], self.particles_state[:, 2], ddof=0)

            vals, vecs = np.linalg.eig(self.position_covariance_matrix)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            self.position_covariance_eigenvalues = np.sqrt(np.maximum(0, vals))
            self.position_covariance_orientation = np.arctan2(
                vecs[1, 0], vecs[0, 0])

        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix calculation failed (singular).")
            self.position_covariance_matrix = np.eye(2) * 1e-6
            self.position_covariance_eigenvalues = np.array([1e-3, 1e-3])
            self.position_covariance_orientation = 0.0

    def evaluate_filter_quality(self, observer_location: Location, measured_range: float):
        """
        Evaluates the filter quality based on range error and dispersion. Can reset initialization flag.

        Args:
            observer_location: The current observer location.
            measured_range: The current measured range.
        """
        if not self.is_initialized:
            return

        max_mean_range_error_factor = self.config.pf_eval_max_mean_range_error_factor
        dispersion_threshold = self.config.pf_eval_dispersion_threshold
        max_mean_range_error = max_mean_range_error_factor * self.max_particle_range

        if self.estimation_method == 'area':
            if np.max(self.weights) < 0.1:
                print(
                    "Filter quality poor (max weight low in area method). Reinitializing.")
                self.is_initialized = False
            return

        if measured_range == -1:
            return

        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        particle_ranges = np.sqrt(dx**2 + dy**2)
        mean_particle_range = np.mean(particle_ranges)
        mean_range_error = abs(mean_particle_range - measured_range)

        confidence_scale = 1.96
        ellipse_axis1 = self.position_covariance_eigenvalues[0] * \
            confidence_scale
        ellipse_axis2 = self.position_covariance_eigenvalues[1] * \
            confidence_scale
        dispersion = np.sqrt(ellipse_axis1**2 + ellipse_axis2**2)

        if mean_range_error > max_mean_range_error and dispersion < dispersion_threshold:
            print(
                f"Filter quality poor (Error: {mean_range_error:.1f} > {max_mean_range_error:.1f}, Dispersion: {dispersion:.1f} < {dispersion_threshold}). Reinitializing.")
            self.is_initialized = False


# --- Target Tracker Class (using Particle Filter) ---

class TrackedTargetPF:
    """
    Represents a target whose state is estimated using a Particle Filter.
    This class acts as an interface to the ParticleFilterCore.
    """

    def __init__(self, config: ParticleFilterConfig):
        """
        Initializes the target tracker with a Particle Filter using configuration.

        Args:
            config: Configuration object for the particle filter.
        """
        self.config = config
        self.resampling_method = config.resampling_method

        self.pf_core = ParticleFilterCore(config=config)

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        self.previous_particles_state: Optional[np.ndarray] = None
        self.current_particles_state: Optional[np.ndarray] = None

        # Least Squares attributes (kept separate from PF core logic)
        self.ls_state_history: List[np.ndarray] = []
        self.ls_observer_x_history: List[float] = []
        self.ls_observer_y_history: List[float] = []
        self.ls_range_history: List[float] = []
        self.ls_estimated_position: Optional[np.ndarray] = None

    def update(self,
               dt: float,
               has_new_range: bool,
               range_measurement: float,
               observer_location: Location,
               perform_update_step: bool = True):
        """
        Performs one cycle of the Particle Filter update (predict, weight, resample).

        Args:
            dt: Time step since the last update.
            has_new_range: Boolean indicating if range_measurement is new data.
            range_measurement: The measured range. Use -1.0 if no new data.
            observer_location: Current location of the observer/sensor.
            perform_update_step: If False, only prediction is done.
        """
        if not perform_update_step and not self.pf_core.is_initialized:
            return

        if not self.pf_core.is_initialized:
            if has_new_range and range_measurement > 0:
                self.pf_core.initialize_particles(
                    observer_location, range_measurement)
                self.current_particles_state = self.pf_core.particles_state.copy()
            else:
                return

        self.previous_particles_state = self.pf_core.particles_state.copy()
        self.pf_core.predict(dt)

        if perform_update_step:
            effective_range = range_measurement if has_new_range else -1.0
            self.pf_core.update_weights(effective_range, observer_location)
            self.pf_core.resample_particles(method=self.resampling_method)
            self.pf_core.evaluate_filter_quality(
                observer_location, effective_range)

        self.pf_core.estimate_target_state(method=2)

        self.estimated_location = self.pf_core.estimated_location
        self.estimated_velocity = self.pf_core.estimated_velocity
        self.current_particles_state = self.pf_core.particles_state.copy()

    def update_least_squares(self, dt: float, has_new_range: bool, range_measurement: float, observer_location: Location, num_points_to_use: int = 30):
        """
        Updates the target estimate using a simple Least Squares batch method.

        Args:
            dt: Time step (used for velocity estimation).
            has_new_range: True if range_measurement is new.
            range_measurement: The measured range.
            observer_location: Current location of the observer.
            num_points_to_use: How many recent points to use in the LS calculation.
        """
        if has_new_range and range_measurement > 0:
            self.ls_range_history.append(range_measurement)
            self.ls_observer_x_history.append(observer_location.x)
            self.ls_observer_y_history.append(observer_location.y)

        n_points = len(self.ls_observer_x_history)
        if n_points < 3:
            if self.ls_state_history:
                last_state = self.ls_state_history[-1].copy()
                if dt > 1e-9:
                    last_state[0] += last_state[1] * dt
                    last_state[2] += last_state[3] * dt
                self.ls_state_history.append(last_state)
            return False

        start_idx = max(0, n_points - num_points_to_use)
        used_x = self.ls_observer_x_history[start_idx:]
        used_y = self.ls_observer_y_history[start_idx:]
        used_z = self.ls_range_history[start_idx:]
        num = len(used_x)

        if num < 3:
            return False

        P = np.array([used_x, used_y])
        A = np.zeros((num, 3))
        b = np.zeros((num, 1))

        A[:, 0] = 2 * P[0, :]
        A[:, 1] = 2 * P[1, :]
        A[:, 2] = -1
        b[:, 0] = P[0, :]**2 + P[1, :]**2 - np.array(used_z)**2

        try:
            solution = np.linalg.pinv(A) @ b
            self.ls_estimated_position = solution[0:2, 0]
        except np.linalg.LinAlgError:
            print('WARNING: LS solution failed (singular matrix). Skipping LS update.')
            if self.ls_state_history:
                last_state = self.ls_state_history[-1].copy()
                if dt > 1e-9:
                    last_state[0] += last_state[1] * dt
                    last_state[2] += last_state[3] * dt
                self.ls_state_history.append(last_state)
            return False

        ls_velocity = np.array([0.0, 0.0])
        ls_orientation = 0.0

        if self.ls_state_history:
            prev_state = self.ls_state_history[-1]
            dx = self.ls_estimated_position[0] - prev_state[0]
            dy = self.ls_estimated_position[1] - prev_state[2]
            if dt > 1e-6:
                ls_velocity[0] = dx / dt
                ls_velocity[1] = dy / dt
            if np.sqrt(dx**2 + dy**2) > 1e-6:
                ls_orientation = np.arctan2(dy, dx)
            else:
                ls_orientation = prev_state[4]

        current_ls_state = np.array([
            self.ls_estimated_position[0], ls_velocity[0],
            self.ls_estimated_position[1], ls_velocity[1],
            ls_orientation
        ])
        self.ls_state_history.append(current_ls_state)
        return True

    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the internal state of the particle filter.
        
        Returns:
            Dictionary containing the serialized state
        """
        state = {
            "is_initialized": self.pf_core.is_initialized,
            "particles_state": self.pf_core.particles_state.tolist() if self.pf_core.particles_state is not None else None,
            "weights": self.pf_core.weights.tolist() if self.pf_core.weights is not None else None,
            "position_covariance_matrix": self.pf_core.position_covariance_matrix.tolist() if hasattr(self.pf_core, "position_covariance_matrix") else None,
            "position_covariance_eigenvalues": self.pf_core.position_covariance_eigenvalues.tolist() if hasattr(self.pf_core, "position_covariance_eigenvalues") else None,
            "position_covariance_orientation": self.pf_core.position_covariance_orientation if hasattr(self.pf_core, "position_covariance_orientation") else 0.0,
        }
        
        # Include estimated location and velocity if available
        if self.estimated_location is not None:
            state["estimated_location"] = (self.estimated_location.x, self.estimated_location.y, self.estimated_location.depth)
        
        if self.estimated_velocity is not None:
            state["estimated_velocity"] = (self.estimated_velocity.x, self.estimated_velocity.y, self.estimated_velocity.z)
            
        # Store observer location
        if self.pf_core.previous_observer_location is not None:
            loc = self.pf_core.previous_observer_location
            state["previous_observer_location"] = (loc.x, loc.y, loc.depth)
            
        return state
    
    def decode_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Restores the internal state of the particle filter.
        
        Args:
            state_dict: Dictionary containing the serialized state
        """
        # Restore initialization flag
        self.pf_core.is_initialized = state_dict.get("is_initialized", False)
        
        # Restore particles and weights
        if "particles_state" in state_dict and state_dict["particles_state"] is not None:
            self.pf_core.particles_state = np.array(state_dict["particles_state"])
            self.current_particles_state = self.pf_core.particles_state.copy()
        
        if "weights" in state_dict and state_dict["weights"] is not None:
            self.pf_core.weights = np.array(state_dict["weights"])
        
        # Restore covariance information
        if "position_covariance_matrix" in state_dict and state_dict["position_covariance_matrix"] is not None:
            self.pf_core.position_covariance_matrix = np.array(state_dict["position_covariance_matrix"])
        
        if "position_covariance_eigenvalues" in state_dict and state_dict["position_covariance_eigenvalues"] is not None:
            self.pf_core.position_covariance_eigenvalues = np.array(state_dict["position_covariance_eigenvalues"])
        
        if "position_covariance_orientation" in state_dict:
            self.pf_core.position_covariance_orientation = state_dict["position_covariance_orientation"]
        
        # Restore estimated location and velocity
        if "estimated_location" in state_dict:
            x, y, depth = state_dict["estimated_location"]
            self.estimated_location = Location(x=x, y=y, depth=depth)
            self.pf_core.estimated_location = self.estimated_location
        else:
            self.estimated_location = None
            self.pf_core.estimated_location = None
            
        if "estimated_velocity" in state_dict:
            x, y, z = state_dict["estimated_velocity"]
            self.estimated_velocity = Velocity(x=x, y=y, z=z)
            self.pf_core.estimated_velocity = self.estimated_velocity
        else:
            self.estimated_velocity = None
            self.pf_core.estimated_velocity = None
        
        # Restore observer location
        if "previous_observer_location" in state_dict:
            x, y, depth = state_dict["previous_observer_location"]
            self.pf_core.previous_observer_location = Location(x=x, y=y, depth=depth)
        else:
            self.pf_core.previous_observer_location = None
