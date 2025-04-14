"""
Particle Filter implementation for target tracking, using Location/Velocity objects.
Refactored for clarity and integration with world_objects.
"""
from typing import List, Optional, Dict, Any
import random
import numpy as np

from world_objects import Location, Velocity
from configs import ParticleFilterConfig

# --- Particle Filter Core Logic ---

class ParticleFilterCore:
    """ Core implementation of the Particle Filter algorithm. """

    def __init__(self, config: ParticleFilterConfig):
        """ Initializes the core particle filter components using configuration. """
        self.state_dimension = 4  # [x, vx, y, vy]
        self.config = config
        self.num_particles = config.num_particles
        # self.estimation_method = config.estimation_method # Removed - only range supported now
        self.max_particle_range = config.max_particle_range

        self.process_noise_position = config.process_noise_pos
        self.process_noise_orientation = config.process_noise_orient
        self.process_noise_velocity = config.process_noise_vel
        self.measurement_noise_stddev = config.measurement_noise_stddev

        self.particles_state = np.zeros((self.num_particles, self.state_dimension))
        # self.previous_particles_state = np.zeros((self.num_particles, self.state_dimension)) # Not used

        self.weights = np.ones(self.num_particles) / self.num_particles

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        self.position_covariance_matrix = np.eye(2)
        self.position_covariance_eigenvalues = np.array([0.02, 0.02])
        self.position_covariance_orientation = 0.0

        self.is_initialized = False
        self.previous_observer_location: Optional[Location] = None

    def initialize_particles(self, observer_location: Location, initial_range_guess: float):
        """ Initializes particle positions and velocities around an initial guess. """
        initial_range_stddev = self.config.initial_range_stddev
        initial_velocity_guess = self.config.initial_velocity_guess

        for i in range(self.num_particles):
            angle = random.uniform(0, 2 * np.pi)
            radius = random.gauss(initial_range_guess, initial_range_stddev)

            self.particles_state[i, 0] = observer_location.x + radius * np.cos(angle) # x
            self.particles_state[i, 2] = observer_location.y + radius * np.sin(angle) # y

            initial_orientation = random.uniform(0, 2 * np.pi)
            velocity_magnitude = abs(random.gauss(initial_velocity_guess, initial_velocity_guess / 2.0 + 1e-6))
            self.particles_state[i, 1] = velocity_magnitude * np.cos(initial_orientation) # vx
            self.particles_state[i, 3] = velocity_magnitude * np.sin(initial_orientation) # vy

        self.weights.fill(1.0 / self.num_particles)
        self.estimate_target_state()
        self.is_initialized = True

    def predict(self, dt: float):
        """ Predicts the next state of each particle based on a simple motion model. """
        if not self.is_initialized:
            return

        for i in range(self.num_particles):
            vx = self.particles_state[i, 1]
            vy = self.particles_state[i, 3]
            current_orientation = np.arctan2(vy, vx)

            orientation_noise = random.uniform(-self.process_noise_orientation, self.process_noise_orientation)
            new_orientation = (current_orientation + orientation_noise) % (2 * np.pi)

            current_velocity_magnitude = np.sqrt(vx**2 + vy**2)
            velocity_noise = random.uniform(-self.process_noise_velocity, self.process_noise_velocity)
            new_velocity_magnitude = max(0.0, current_velocity_magnitude + velocity_noise)

            distance_travelled = current_velocity_magnitude * dt
            position_noise = random.uniform(-self.process_noise_position, self.process_noise_position)
            effective_distance = distance_travelled + position_noise

            self.particles_state[i, 0] += effective_distance * np.cos(new_orientation) # Update x
            self.particles_state[i, 2] += effective_distance * np.sin(new_orientation) # Update y
            self.particles_state[i, 1] = new_velocity_magnitude * np.cos(new_orientation) # Update vx
            self.particles_state[i, 3] = new_velocity_magnitude * np.sin(new_orientation) # Update vy

    def _calculate_likelihood(self, predicted_range: float, measured_range: float) -> float:
        """ Calculates the likelihood weight based on range measurement. """
        if measured_range <= 0: # Handle invalid or missing measurements
            return 1.0 # Assign uniform likelihood if no valid measurement

        noise_stddev = self.measurement_noise_stddev
        variance = noise_stddev ** 2
        if variance < 1e-9: variance = 1e-9 # Prevent division by zero

        # Gaussian likelihood
        exponent = -((predicted_range - measured_range) ** 2) / (2 * variance)
        # Denominator sqrt(2*pi*var) is constant for all particles, can be omitted for weighting
        # likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * variance)
        likelihood = np.exp(exponent)
        return likelihood + 1e-9 # Add epsilon to prevent zero weights

    def update_weights(self, measured_range: float, observer_location: Location):
        """ Updates particle weights based on the latest measurement. """
        if not self.is_initialized:
            return

        self.previous_observer_location = observer_location

        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        predicted_ranges = np.sqrt(dx**2 + dy**2) # 2D range prediction

        for i in range(self.num_particles):
            likelihood = self._calculate_likelihood(predicted_ranges[i], measured_range)
            self.weights[i] *= likelihood

        total_weight = np.sum(self.weights)
        if total_weight > 1e-9:
            self.weights /= total_weight
        else:
            # print("Warning: Particle weights sum near zero. Reinitializing weights uniformly.")
            self.weights.fill(1.0 / self.num_particles)

    def resample_particles(self, method: int):
        """ Resamples particles based on their weights to combat particle degeneracy. """
        if not self.is_initialized:
            return

        new_particles_state = np.zeros_like(self.particles_state)
        n = self.num_particles
        initial_velocity_guess = self.config.initial_velocity_guess # Used for random injection

        neff = 1.0 / np.sum(self.weights**2) if np.sum(self.weights**2) > 0 else 0
        # print(f"Effective Sample Size (Neff): {neff:.2f}")
        # Resample only if Neff is below a threshold (e.g., N/2)
        resample_threshold = n / 2.0
        if neff >= resample_threshold:
             # print("Skipping resampling, Neff above threshold.")
             return # Don't resample if Neff is high enough

        # --- Systematic Resampling (Method 2) ---
        # Generally preferred for efficiency and low variance
        cumulative_sum = np.cumsum(self.weights)
        step = 1.0 / n
        u = random.uniform(0, step)
        i = 0
        for j in range(n):
            while i < n and u > cumulative_sum[i]: # Ensure i doesn't exceed bounds
                i += 1
            idx_to_copy = min(i, n - 1) # Clamp index to valid range
            new_particles_state[j, :] = self.particles_state[idx_to_copy, :]
            u += step

        # Methods 1, 3, 3.2 removed for simplicity, focusing on systematic resampling
        # Method 1 (Multinomial) can have higher variance
        # Method 3/3.2 (Hybrid) adds complexity

        self.particles_state = new_particles_state
        self.weights.fill(1.0 / self.num_particles) # Reset weights after resampling

    def estimate_target_state(self, method: int = 2):
        """ Estimates the target state (location and velocity) using weighted mean. """
        if not self.is_initialized or self.num_particles == 0:
            self.estimated_location = None
            self.estimated_velocity = None
            return

        total_weight = np.sum(self.weights)
        if total_weight < 1e-9: # Handle case where weights are near zero
            mean_state = np.mean(self.particles_state, axis=0)
        else:
            mean_state = np.average(self.particles_state, axis=0, weights=self.weights)

        self.estimated_location = Location(x=mean_state[0], y=mean_state[2], depth=0.0) # Assuming 2D tracking
        self.estimated_velocity = Velocity(x=mean_state[1], y=mean_state[3], z=0.0)

        try:
            # Calculate weighted covariance
            if total_weight > 1e-9:
                x_coords = self.particles_state[:, 0]
                y_coords = self.particles_state[:, 2]
                avg_x = self.estimated_location.x
                avg_y = self.estimated_location.y
                # Weighted covariance calculation
                cov_xx = np.sum(self.weights * (x_coords - avg_x)**2)
                cov_yy = np.sum(self.weights * (y_coords - avg_y)**2)
                cov_xy = np.sum(self.weights * (x_coords - avg_x) * (y_coords - avg_y))
                self.position_covariance_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            else: # Fallback to unweighted covariance if weights are zero
                self.position_covariance_matrix = np.cov(
                    self.particles_state[:, 0], self.particles_state[:, 2], ddof=0)

            # Calculate eigenvalues and orientation for uncertainty ellipse
            vals, vecs = np.linalg.eigh(self.position_covariance_matrix) # Use eigh for symmetric matrix
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            self.position_covariance_eigenvalues = np.sqrt(np.maximum(0, vals)) # Ensure non-negative before sqrt
            self.position_covariance_orientation = np.arctan2(vecs[1, 0], vecs[0, 0])

        except np.linalg.LinAlgError:
            # print("Warning: Covariance matrix calculation failed (singular).")
            self.position_covariance_matrix = np.eye(2) * 1e-6
            self.position_covariance_eigenvalues = np.array([1e-3, 1e-3])
            self.position_covariance_orientation = 0.0

    def evaluate_filter_quality(self, observer_location: Location, measured_range: float):
        """ Evaluates filter quality. Can reset initialization flag if quality is poor. """
        if not self.is_initialized or measured_range <= 0:
            return # Cannot evaluate without initialization or valid measurement

        max_mean_range_error_factor = self.config.pf_eval_max_mean_range_error_factor
        dispersion_threshold = self.config.pf_eval_dispersion_threshold
        max_mean_range_error = max_mean_range_error_factor * self.max_particle_range

        # Calculate mean range error
        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        particle_ranges = np.sqrt(dx**2 + dy**2)
        # Use weighted mean range if weights are valid
        total_weight = np.sum(self.weights)
        if total_weight > 1e-9:
             mean_particle_range = np.average(particle_ranges, weights=self.weights)
        else:
             mean_particle_range = np.mean(particle_ranges)

        mean_range_error = abs(mean_particle_range - measured_range)

        # Calculate dispersion (using covariance eigenvalues)
        confidence_scale = 1.96 # For approx 95% confidence interval
        ellipse_axis1 = self.position_covariance_eigenvalues[0] * confidence_scale
        ellipse_axis2 = self.position_covariance_eigenvalues[1] * confidence_scale
        dispersion = np.sqrt(ellipse_axis1**2 + ellipse_axis2**2) # Geometric mean or similar measure? Using sqrt sum squares for now.

        # Check quality criteria
        # Condition: Large range error AND low dispersion (filter converged to wrong place)
        if mean_range_error > max_mean_range_error and dispersion < dispersion_threshold:
            # print(f"Filter quality poor (Error: {mean_range_error:.1f} > {max_mean_range_error:.1f}, "
            #       f"Dispersion: {dispersion:.1f} < {dispersion_threshold}). Reinitializing.")
            self.is_initialized = False # Trigger re-initialization on next valid measurement


# --- Target Tracker Class (using Particle Filter) ---

class TrackedTargetPF:
    """ Represents a target whose state is estimated using a Particle Filter. """

    def __init__(self, config: ParticleFilterConfig):
        """ Initializes the target tracker with a Particle Filter. """
        self.config = config
        self.resampling_method = config.resampling_method # Currently only systematic (2) is implemented in core
        self.pf_core = ParticleFilterCore(config=config)

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        # Keep track of particle states mainly for visualization/debugging if needed
        self.previous_particles_state: Optional[np.ndarray] = None # State before predict
        self.current_particles_state: Optional[np.ndarray] = None # State after resample

        # Attributes for other estimation methods (like LS) removed from this class

    def update(self,
               dt: float,
               has_new_range: bool,
               range_measurement: float,
               observer_location: Location,
               perform_update_step: bool = True):
        """ Performs one cycle of the Particle Filter update (predict, weight, resample). """

        # Step 1: Initialization check
        if not self.pf_core.is_initialized:
            if has_new_range and range_measurement > 0:
                # print(f"PF Initializing: Obs={observer_location}, Range={range_measurement:.2f}")
                self.pf_core.initialize_particles(observer_location, range_measurement)
                if self.pf_core.is_initialized:
                    self.current_particles_state = self.pf_core.particles_state.copy()
                    self.estimated_location = self.pf_core.estimated_location
                    self.estimated_velocity = self.pf_core.estimated_velocity
                return # Initialized, done for this step
            else:
                return # Cannot initialize without a valid range measurement

        # If initialized, proceed with predict/update cycle

        # Store state before prediction (optional, for debugging/analysis)
        self.previous_particles_state = self.pf_core.particles_state.copy()

        # Step 2: Prediction
        self.pf_core.predict(dt)

        # Step 3: Update (Weighting, Resampling, Evaluation) - only if requested
        if perform_update_step:
            effective_range = range_measurement if has_new_range else -1.0 # Use -1 if no new range
            self.pf_core.update_weights(effective_range, observer_location)
            self.pf_core.resample_particles(method=self.resampling_method)
            # Evaluate filter quality based on the *new* measurement
            self.pf_core.evaluate_filter_quality(observer_location, effective_range)

            # If filter quality check reset initialization, stop here
            if not self.pf_core.is_initialized:
                 # print("PF re-initialization triggered by quality check.")
                 self.estimated_location = None
                 self.estimated_velocity = None
                 self.current_particles_state = None
                 self.previous_particles_state = None
                 return

        # Step 4: Estimation (always estimate state after predict/update)
        self.pf_core.estimate_target_state(method=2) # Use weighted mean

        # Update public attributes
        self.estimated_location = self.pf_core.estimated_location
        self.estimated_velocity = self.pf_core.estimated_velocity
        self.current_particles_state = self.pf_core.particles_state.copy()


    def encode_state(self) -> Dict[str, Any]:
        """ Encodes the internal state of the particle filter. """
        state = {
            "config_dump": self.config.model_dump(), # Store config used
            "is_initialized": self.pf_core.is_initialized,
            "particles_state": self.pf_core.particles_state.tolist() if self.pf_core.particles_state is not None else None,
            "weights": self.pf_core.weights.tolist() if self.pf_core.weights is not None else None,
            # Covariance info useful for analysis/visualization
            "position_covariance_matrix": self.pf_core.position_covariance_matrix.tolist() if hasattr(self.pf_core, "position_covariance_matrix") else None,
            "position_covariance_eigenvalues": self.pf_core.position_covariance_eigenvalues.tolist() if hasattr(self.pf_core, "position_covariance_eigenvalues") else None,
            "position_covariance_orientation": self.pf_core.position_covariance_orientation if hasattr(self.pf_core, "position_covariance_orientation") else 0.0,
        }

        if self.estimated_location is not None:
            loc = self.estimated_location
            state["estimated_location"] = (loc.x, loc.y, loc.depth)
        if self.estimated_velocity is not None:
            vel = self.estimated_velocity
            state["estimated_velocity"] = (vel.x, vel.y, vel.z)
        if self.pf_core.previous_observer_location is not None:
            loc = self.pf_core.previous_observer_location
            state["previous_observer_location"] = (loc.x, loc.y, loc.depth)

        return state

    def decode_state(self, state_dict: Dict[str, Any]) -> None:
        """ Restores the internal state of the particle filter. """
        # Config check could be added here if needed
        self.pf_core.is_initialized = state_dict.get("is_initialized", False)

        if "particles_state" in state_dict and state_dict["particles_state"] is not None:
            self.pf_core.particles_state = np.array(state_dict["particles_state"])
            self.current_particles_state = self.pf_core.particles_state.copy() # Update current state view
        else: # If no particles in state, ensure core reflects this
             self.pf_core.particles_state = np.zeros((self.config.num_particles, self.pf_core.state_dimension))
             self.current_particles_state = None

        if "weights" in state_dict and state_dict["weights"] is not None:
            self.pf_core.weights = np.array(state_dict["weights"])
        else: # Reset weights if not in state
             self.pf_core.weights = np.ones(self.config.num_particles) / self.config.num_particles

        # Restore covariance info if present
        if "position_covariance_matrix" in state_dict and state_dict["position_covariance_matrix"] is not None:
            self.pf_core.position_covariance_matrix = np.array(state_dict["position_covariance_matrix"])
        if "position_covariance_eigenvalues" in state_dict and state_dict["position_covariance_eigenvalues"] is not None:
            self.pf_core.position_covariance_eigenvalues = np.array(state_dict["position_covariance_eigenvalues"])
        if "position_covariance_orientation" in state_dict:
            self.pf_core.position_covariance_orientation = state_dict["position_covariance_orientation"]

        # Restore estimates
        if "estimated_location" in state_dict:
            x, y, depth = state_dict["estimated_location"]
            self.estimated_location = Location(x=x, y=y, depth=depth)
            self.pf_core.estimated_location = self.estimated_location # Sync core estimate
        else:
            self.estimated_location = None
            self.pf_core.estimated_location = None

        if "estimated_velocity" in state_dict:
            x, y, z = state_dict["estimated_velocity"]
            self.estimated_velocity = Velocity(x=x, y=y, z=z)
            self.pf_core.estimated_velocity = self.estimated_velocity # Sync core estimate
        else:
            self.estimated_velocity = None
            self.pf_core.estimated_velocity = None

        # Restore previous observer location
        if "previous_observer_location" in state_dict:
            x, y, depth = state_dict["previous_observer_location"]
            self.pf_core.previous_observer_location = Location(x=x, y=y, depth=depth)
        else:
            self.pf_core.previous_observer_location = None