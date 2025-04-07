"""
Particle Filter implementation for target tracking, using Location/Velocity objects.
Original Author: Ivan Masmitja Rusinol (March 29, 2020)
Refactored for clarity and integration with world_objects.
Project: AIforUTracking / Refactoring Exercise
"""

import numpy as np
import random
import torch
from typing import List, Optional
from world_objects import Location, Velocity

# --- Constants ---
SOUND_SPEED = 1500.0 

# --- Particle Filter Core Logic ---

class ParticleFilterCore:
    """ Core implementation of the Particle Filter algorithm. """

    def __init__(self,
                 initial_range_stddev: float,
                 initial_velocity_guess: float,
                 state_dimension: int = 4, # [x, vx, y, vy]
                 num_particles: int = 6000,
                 estimation_method: str = 'range', # 'range' or 'area'
                 max_particle_range: float = 250.0,
                 device: str = None):
        """
        Initializes the core particle filter components.

        Args:
            initial_range_stddev: Standard deviation for initial particle position spread (range method).
            initial_velocity_guess: Mean initial velocity for particles.
            state_dimension: Dimension of the state vector for each particle (default 4).
            num_particles: Number of particles to use.
            estimation_method: Method for likelihood calculation ('range' or 'area').
            max_particle_range: Maximum range used in 'area' method or for initial spread.
            device: Device to use for tensor operations ('cuda', 'cuda:0', 'cpu', etc).
                   If None, automatically selects CUDA if available, otherwise CPU.
        """
        if state_dimension != 4:
            raise ValueError("State dimension must be 4 ([x, vx, y, vy]) for current implementation.")

        # Set device for PyTorch tensors
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.initial_range_stddev = initial_range_stddev
        self.initial_velocity_guess = initial_velocity_guess
        self.num_particles = num_particles
        self.state_dimension = state_dimension
        self.estimation_method = estimation_method
        self.max_particle_range = max_particle_range # Used in 'area' method and init

        # Particle states: rows are particles, columns are [x, vx, y, vy]
        self.particles_state = torch.zeros((self.num_particles, self.state_dimension), device=self.device)
        self.previous_particles_state = torch.zeros((self.num_particles, self.state_dimension), device=self.device)

        # Particle weights
        self.weights = torch.ones(self.num_particles, device=self.device) / self.num_particles # Initialize normalized

        # Noise parameters (can be set later via set_noise)
        self.process_noise_position = 0.0 # Corresponds to former forward_noise
        self.process_noise_orientation = 0.0 # Corresponds to former turn_noise
        self.process_noise_velocity = 0.0
        self.measurement_noise_stddev = 0.0 # Corresponds to former sense_noise (std dev for range)

        # Internal state estimation result (mean) - Store as Location/Velocity
        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        # Covariance of the estimated position
        self.position_covariance_matrix = torch.eye(2, device=self.device) # Initialize to identity
        self.position_covariance_eigenvalues = torch.tensor([0.02, 0.02], device=self.device) # For ellipse drawing
        self.position_covariance_orientation = 0.0 # Angle of major axis

        # Flag for initialization state
        self.is_initialized = False

        # --- Variables potentially needed for specific likelihood/resampling methods ---
        # Store previous observer location if needed by resampling method 3.2
        self.previous_observer_location: Optional[Location] = None

    def set_noise(self,
                  process_noise_position: float,
                  process_noise_orientation: float,
                  process_noise_velocity: float,
                  measurement_noise_stddev: float):
        """ Sets the noise parameters for the filter. """
        self.process_noise_position = process_noise_position
        self.process_noise_orientation = process_noise_orientation
        self.process_noise_velocity = process_noise_velocity
        self.measurement_noise_stddev = measurement_noise_stddev

    def initialize_particles(self, observer_location: Location, initial_range_guess: float):
        """
        Initializes particle positions and velocities around an initial guess.

        Args:
            observer_location: The initial location of the observer/sensor.
            initial_range_guess: An initial guess for the range to the target.
        """
        for i in range(self.num_particles):
            # Random angle
            angle = random.uniform(0, 2 * np.pi)

            # Random radius based on method
            if self.estimation_method == 'area':
                # Uniform distribution within a disk
                radius = random.uniform(-self.max_particle_range, self.max_particle_range)
            else: # 'range' method
                 # Gaussian-like spread around the initial range guess
                radius = random.gauss(initial_range_guess, self.initial_range_stddev)

            # Initial position (x, y)
            self.particles_state[i, 0] = observer_location.x + radius * np.cos(angle) # x
            self.particles_state[i, 2] = observer_location.y + radius * np.sin(angle) # y

            # Initial velocity (vx, vy)
            initial_orientation = random.uniform(0, 2 * np.pi)
            # Ensure velocity isn't exactly zero if guess is non-zero
            velocity_magnitude = abs(random.gauss(self.initial_velocity_guess, self.initial_velocity_guess / 2.0 + 1e-6))
            self.particles_state[i, 1] = velocity_magnitude * np.cos(initial_orientation) # vx
            self.particles_state[i, 3] = velocity_magnitude * np.sin(initial_orientation) # vy

        self.weights.fill(1.0 / self.num_particles) # Reset weights to uniform
        self.estimate_target_state() # Calculate initial estimate
        self.is_initialized = True

    def predict(self, dt: float):
        """
        Predicts the next state of each particle based on a simple motion model.

        Args:
            dt: Time step interval.
        """
        if not self.is_initialized:
            return # Cannot predict if not initialized

        # Use uniform noise for simplicity, matching original code's non-Gauss path
        use_gaussian_noise = False

        for i in range(self.num_particles):
            # --- Orientation Update ---
            vx = self.particles_state[i, 1]
            vy = self.particles_state[i, 3]
            current_orientation = np.arctan2(vy, vx)

            if use_gaussian_noise:
                orientation_noise = random.gauss(0.0, self.process_noise_orientation)
            else:
                orientation_noise = random.uniform(-self.process_noise_orientation, self.process_noise_orientation)

            new_orientation = (current_orientation + orientation_noise) % (2 * np.pi)

            # --- Velocity Magnitude Update ---
            current_velocity_magnitude = np.sqrt(vx**2 + vy**2)

            if use_gaussian_noise:
                velocity_noise = random.gauss(0.0, self.process_noise_velocity)
            else:
                velocity_noise = random.uniform(-self.process_noise_velocity, self.process_noise_velocity)

            new_velocity_magnitude = max(0.0, current_velocity_magnitude + velocity_noise) # Prevent negative speed

            # --- Position Update ---
            distance_travelled = current_velocity_magnitude * dt

            if use_gaussian_noise:
                position_noise = random.gauss(0.0, self.process_noise_position)
            else:
                position_noise = random.uniform(-self.process_noise_position, self.process_noise_position)

            effective_distance = distance_travelled + position_noise

            # Update particle state
            self.particles_state[i, 0] += effective_distance * np.cos(new_orientation) # x
            self.particles_state[i, 2] += effective_distance * np.sin(new_orientation) # y
            self.particles_state[i, 1] = new_velocity_magnitude * np.cos(new_orientation) # vx
            self.particles_state[i, 3] = new_velocity_magnitude * np.sin(new_orientation) # vy


    def _calculate_likelihood(self,
                              predicted_range: float,
                              measured_range: float,
                              noise_stddev: float) -> float:
        """
        Calculates the likelihood weight based on the estimation method.

        Args:
            predicted_range: The range predicted by a particle.
            measured_range: The actual measured range (-1 if no measurement).
            noise_stddev: The standard deviation of the measurement noise (for 'range' method).

        Returns:
            The likelihood weight for the particle.
        """
        if self.estimation_method == 'area':
            # Area method: Simplified view - is particle inside/outside detection range?
            # This uses Cauchy-like functions based on original comments
            # Consider simplifying this logic if possible.
            sigma_area = 1.0 # Sharpness parameter (was 1, sometimes 40)

            if measured_range != -1: # Ping received: favor particles *inside* max_particle_range
                # High weight if predicted_range < max_particle_range
                return (0.5) - (1.0 / np.pi) * np.arctan((predicted_range - self.max_particle_range) / sigma_area)
            else: # No ping: favor particles *outside* max_particle_range
                 # High weight if predicted_range > max_particle_range
                sigma_area = 40.0 # Less sharp transition for no-ping case
                return (0.5) + (1.0 / np.pi) * np.arctan((predicted_range - self.max_particle_range) / sigma_area)

        elif self.estimation_method == 'range':
            # Range method: Gaussian likelihood based on range difference
            if measured_range == -1:
                # No measurement, cannot update weights based on range.
                # Return uniform likelihood (or handle differently if needed)
                return 1.0 # All particles equally likely given no new info

            # Gaussian PDF calculation
            variance = noise_stddev ** 2
            if variance < 1e-9: # Avoid division by zero
                variance = 1e-9
            exponent = -((predicted_range - measured_range) ** 2) / (2 * variance)
            denominator = np.sqrt(2 * np.pi * variance)
            likelihood = np.exp(exponent) / denominator
            # Add small epsilon to prevent zero weights causing issues in resampling
            return likelihood + 1e-9

        else:
            raise ValueError(f"Unknown estimation_method: {self.estimation_method}")

    def update_weights(self, measured_range: float, observer_location: Location):
        """
        Updates particle weights based on the latest measurement.

        Args:
            measured_range: The measured range to the target (-1 if no new measurement).
            observer_location: The location of the observer/sensor at the time of measurement.
        """
        if not self.is_initialized:
            return

        # Store observer location if needed for resampling method 3.2
        self.previous_observer_location = observer_location

        # Calculate predicted range for each particle (2D distance)
        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        predicted_ranges = np.sqrt(dx**2 + dy**2)

        # Calculate likelihood for each particle
        for i in range(self.num_particles):
            likelihood = self._calculate_likelihood(predicted_ranges[i],
                                                    measured_range,
                                                    self.measurement_noise_stddev)
            # Update weight (multiplicative update)
            self.weights[i] *= likelihood

        # Normalize weights
        total_weight = np.sum(self.weights)
        if total_weight > 1e-9: # Avoid division by zero
            self.weights /= total_weight
        else:
            # Handle case of all weights becoming near zero (e.g., bad measurement, model divergence)
            # Reinitialize weights to uniform? Or keep them as they are?
            # Uniform reinitialization is safer against complete filter collapse.
            print("Warning: Particle weights sum near zero. Reinitializing weights uniformly.")
            self.weights.fill(1.0 / self.num_particles)

    def resample_particles(self, method: int = 2):
        """
        Resamples particles based on their weights to combat particle degeneracy.

        Args:
            method: The resampling algorithm to use.
                    1: Original multinomial resampling (prone to loss of diversity).
                    2: Systematic resampling (generally preferred).
                    3: Systematic + random injection (OCEANS'18 method).
                    3.2: Systematic + observer-centered random injection (TAG-Only mod).
        """
        if not self.is_initialized:
            return

        new_particles_state = np.zeros_like(self.particles_state)
        n = self.num_particles

        if method == 1: # Multinomial Resampling (via resampling wheel)
            index = int(random.random() * n)
            beta = 0.0
            max_weight = np.max(self.weights)
            for i in range(n):
                beta += random.random() * 2.0 * max_weight
                while beta > self.weights[index]:
                    beta -= self.weights[index]
                    index = (index + 1) % n
                new_particles_state[i, :] = self.particles_state[index, :]

        elif method == 2: # Systematic Resampling
            cumulative_sum = np.cumsum(self.weights)
            step = 1.0 / n
            u = random.uniform(0, step)
            i = 0
            for j in range(n):
                while u > cumulative_sum[i]:
                    i += 1
                new_particles_state[j, :] = self.particles_state[i, :]
                u += step

        elif method == 3 or method == 3.2: # Hybrid: Systematic + Random Injection
            # Determine number of particles to replace randomly
            # Ratios seem dependent on particle count, which isn't ideal.
            # Let's use a fixed percentage, e.g., 5%
            random_injection_ratio = 0.05
            num_random = int(n * random_injection_ratio)
            num_systematic = n - num_random

            # --- Systematic Resampling Part ---
            if num_systematic > 0:
                temp_particles_systematic = np.zeros((num_systematic, self.state_dimension))
                cumulative_sum = np.cumsum(self.weights)
                # Adjust step for the number being drawn systematically
                # Use cumulative_sum[-1] in case weights weren't perfectly normalized to 1
                effective_total_weight = cumulative_sum[-1] if cumulative_sum[-1] > 1e-9 else 1.0
                step = effective_total_weight / num_systematic
                u = random.uniform(0, step)
                i = 0
                current_weight_sum = 0.0
                for j in range(num_systematic):
                    # Need to handle potential floating point issues if weights sum slightly != 1
                    target_weight = u + j * step
                    # Iterate through particles until cumulative weight exceeds target
                    while i < n -1 and current_weight_sum + self.weights[i] < target_weight:
                         current_weight_sum += self.weights[i]
                         i += 1
                    temp_particles_systematic[j, :] = self.particles_state[i, :]

                new_particles_state[:num_systematic, :] = temp_particles_systematic

            # --- Random Injection Part ---
            center_x, center_y, injection_radius = None, None, None
            injection_possible = False
            if num_random > 0:
                 # Determine center and radius for random injection
                if method == 3 and self.estimated_location is not None:
                    # Center around the current estimate (OCEANS'18)
                    center_x = self.estimated_location.x
                    center_y = self.estimated_location.y
                    injection_radius = 0.2 # Fixed radius from original code
                    injection_possible = True
                elif method == 3.2 and self.previous_observer_location is not None:
                     # Center around the *previous* observer location (TAG-Only mod)
                    center_x = self.previous_observer_location.x
                    center_y = self.previous_observer_location.y
                    injection_radius = self.max_particle_range # Radius from original code
                    injection_possible = True

                if not injection_possible:
                    # Fallback: If estimate/previous observer is not available, inject randomly in domain?
                    # Or skip injection? Let's skip if center is undefined.
                    print("Warning: Cannot perform random injection for resampling method", method)
                    # Copy remaining particles if skipping injection
                    if num_random > 0 and num_systematic > 0:
                        # If systematic sampling happened, just copy the last few best particles
                         indices_to_copy = np.argsort(self.weights)[-num_random:]
                         new_particles_state[num_systematic:, :] = self.particles_state[indices_to_copy, :]
                    elif num_random > 0: # If only random injection was supposed to happen
                         # Just keep the original particles? Or reinitialize randomly?
                         # Keeping original seems safest if systematic failed or wasn't done
                         new_particles_state[num_systematic:, :] = self.particles_state[num_systematic:, :] # Assuming indices align

                else: # Injection is possible
                    for k in range(num_random):
                        idx = num_systematic + k
                        # Random position within a disk
                        angle = random.uniform(0, 2 * np.pi)
                        radius = random.uniform(0, injection_radius) # Uniform radius sampling
                        # Alternative: sample radius^2 uniformly for uniform area distribution
                        # radius = np.sqrt(random.uniform(0, injection_radius**2))
                        new_particles_state[idx, 0] = center_x + radius * np.cos(angle) # x
                        new_particles_state[idx, 2] = center_y + radius * np.sin(angle) # y

                        # Random velocity (same as initialization)
                        orientation = random.uniform(0, 2 * np.pi)
                        velocity_magnitude = abs(random.gauss(self.initial_velocity_guess, self.initial_velocity_guess / 2.0 + 1e-6))
                        new_particles_state[idx, 1] = velocity_magnitude * np.cos(orientation) # vx
                        new_particles_state[idx, 3] = velocity_magnitude * np.sin(orientation) # vy

        else:
            raise ValueError(f"Unsupported resampling method: {method}")

        self.particles_state = new_particles_state
        # Reset weights to uniform after resampling
        self.weights.fill(1.0 / self.num_particles)

    def estimate_target_state(self, method: int = 2):
        """
        Estimates the target state (location and velocity) from the particles.

        Args:
            method: 1 for simple mean, 2 for weighted mean.
        """
        if not self.is_initialized or self.num_particles == 0:
            self.estimated_location = None
            self.estimated_velocity = None
            return

        mean_state = np.zeros(self.state_dimension)
        total_weight = np.sum(self.weights)

        if method == 1: # Simple Average
            mean_state = np.mean(self.particles_state, axis=0)
        elif method == 2: # Weighted Average
            if total_weight < 1e-9: # Avoid division by zero if weights collapsed
                 mean_state = np.mean(self.particles_state, axis=0) # Fallback to simple average
            else:
                mean_state = np.average(self.particles_state, axis=0, weights=self.weights)
        else:
             raise ValueError(f"Unsupported estimation method: {method}")

        # Store result as Location and Velocity objects
        # Assuming PF is 2D, set depth/z_velocity to 0.
        self.estimated_location = Location(x=mean_state[0], y=mean_state[2], depth=0.0)
        self.estimated_velocity = Velocity(x=mean_state[1], y=mean_state[3], z=0.0)

        # --- Covariance Calculation ---
        # Calculate covariance of particle positions [x, y]
        try:
            if method == 2 and total_weight > 1e-9:
                 # Weighted covariance calculation
                 x_coords = self.particles_state[:, 0]
                 y_coords = self.particles_state[:, 2]
                 avg_x = self.estimated_location.x
                 avg_y = self.estimated_location.y
                 # ddof=0 for population covariance estimate based on weighted sample
                 cov_xx = np.sum(self.weights * (x_coords - avg_x)**2) # / total_weight # Normalized weights sum to 1 approx.
                 cov_yy = np.sum(self.weights * (y_coords - avg_y)**2) # / total_weight
                 cov_xy = np.sum(self.weights * (x_coords - avg_x) * (y_coords - avg_y)) # / total_weight
                 self.position_covariance_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            else:
                 # Unweighted covariance (or if weights collapsed)
                 # Use ddof=0 if considering particles as the full population
                 self.position_covariance_matrix = np.cov(self.particles_state[:, 0], self.particles_state[:, 2], ddof=0)

            # Eigen decomposition for ellipse properties
            vals, vecs = np.linalg.eig(self.position_covariance_matrix)
            # Sort eigenvalues and eigenvectors (largest first)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            self.position_covariance_eigenvalues = np.sqrt(np.maximum(0, vals)) # Ensure non-negative before sqrt
            # Angle of the first eigenvector (corresponding to largest eigenvalue)
            self.position_covariance_orientation = np.arctan2(vecs[1, 0], vecs[0, 0])

        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix calculation failed (singular).")
            self.position_covariance_matrix = np.eye(2) * 1e-6 # Set to small diagonal matrix
            self.position_covariance_eigenvalues = np.array([1e-3, 1e-3])
            self.position_covariance_orientation = 0.0


    def evaluate_filter_quality(self, observer_location: Location, measured_range: float, max_mean_range_error: float = 50.0):
        """
        Evaluates the filter quality. Can reset initialization flag if poor.

        Args:
            observer_location: The current observer location.
            measured_range: The current measured range.
            max_mean_range_error: Threshold for mean particle range error to trigger re-init.
        """
        if not self.is_initialized:
             return

        if self.estimation_method == 'area':
             # Check based on max weight for 'area' method
            if np.max(self.weights) < 0.1:
                 print("Filter quality poor (max weight low in area method). Reinitializing.")
                 self.is_initialized = False
            return # Don't do range error check for area method

        if measured_range == -1:
            # Cannot evaluate range error without a valid measurement
            return

        # --- Evaluation for 'range' method ---
        # Calculate mean range error of particles
        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        particle_ranges = np.sqrt(dx**2 + dy**2)
        mean_particle_range = np.mean(particle_ranges) # Using simple mean as per original check logic
        mean_range_error = abs(mean_particle_range - measured_range)

        # Calculate dispersion based on covariance (optional)
        # Using eigenvalues calculated in estimate_target_state
        # Confidence interval scaling factor (e.g., 95% CI uses approx 1.96 std dev)
        confidence_scale = 1.96 # For 95% CI ellipse axes lengths
        ellipse_axis1 = self.position_covariance_eigenvalues[0] * confidence_scale
        ellipse_axis2 = self.position_covariance_eigenvalues[1] * confidence_scale
        # Dispersion metric (e.g., geometric mean or RMS of axes)
        dispersion = np.sqrt(ellipse_axis1**2 + ellipse_axis2**2)

        # Condition to reset initialization (from original code)
        # If mean range error is large BUT dispersion is small (filter converged wrongly?)
        if mean_range_error > max_mean_range_error and dispersion < 5.0:
            print(f"Filter quality poor (Error: {mean_range_error:.1f} > {max_mean_range_error}, Dispersion: {dispersion:.1f} < 5). Reinitializing.")
            self.is_initialized = False


# --- Target Tracker Class (using Particle Filter) ---

class TrackedTargetPF:
    """
    Represents a target whose state is estimated using a Particle Filter.
    This class acts as an interface to the ParticleFilterCore.
    """
    def __init__(self,
                 initial_range_stddev: float = 0.02,
                 initial_velocity_guess: float = 0.2,
                 num_particles: int = 1000,
                 estimation_method: str = 'range', # 'range' or 'area'
                 max_particle_range: float = 250.0,
                 process_noise_pos: float = 0.01,
                 process_noise_orient: float = 0.1,
                 process_noise_vel: float = 0.01,
                 measurement_noise_stddev: float = 5.0,  # Increase from 0.005 to 5.0
                 resampling_method: int = 2):
        """
        Initializes the target tracker with a Particle Filter.

        Args:
            initial_range_stddev: Std dev for initial particle position spread.
            initial_velocity_guess: Mean initial velocity guess for particles.
            num_particles: Number of particles.
            estimation_method: 'range' or 'area'.
            max_particle_range: Max range for 'area' method or initialization.
            process_noise_pos: Process noise for position prediction.
            process_noise_orient: Process noise for orientation prediction.
            process_noise_vel: Process noise for velocity prediction.
            measurement_noise_stddev: Std dev of range measurement noise.
            resampling_method: Integer code for resampling algorithm.
        """
        self.method = estimation_method # Store for potential external access
        self.resampling_method = resampling_method

        # Create the particle filter core instance
        self.pf_core = ParticleFilterCore(
            initial_range_stddev=initial_range_stddev,
            initial_velocity_guess=initial_velocity_guess,
            state_dimension=4,
            num_particles=num_particles,
            estimation_method=estimation_method,
            max_particle_range=max_particle_range
        )

        # Set noise parameters
        self.pf_core.set_noise(
            process_noise_position=process_noise_pos,
            process_noise_orientation=process_noise_orient,
            process_noise_velocity=process_noise_vel,
            measurement_noise_stddev=measurement_noise_stddev
        )

        # --- Publicly accessible estimated state ---
        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None

        # --- Store particle states for visualization/debugging (optional) ---
        self.previous_particles_state: Optional[np.ndarray] = None
        self.current_particles_state: Optional[np.ndarray] = None

        # --- Least Squares attributes (Kept from original, but separated from PF logic) ---
        # Consider moving LS to a separate class if used extensively.
        self.ls_state_history: List[np.ndarray] = [] # Stores [x, vx, y, vy, orientation] from LS
        self.ls_observer_x_history: List[float] = []
        self.ls_observer_y_history: List[float] = []
        self.ls_range_history: List[float] = []
        self.ls_estimated_position: Optional[np.ndarray] = None # Stores [x, y] from LS calc

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
            range_measurement: The measured range. Use -1.0 if no new data (has_new_range is False).
            observer_location: Current location of the observer/sensor.
            perform_update_step: If False, only prediction is done.
        """
        if not perform_update_step and not self.pf_core.is_initialized:
             # If not performing update and not initialized, do nothing
             return

        # 1. Initialize if necessary
        if not self.pf_core.is_initialized:
            if has_new_range and range_measurement > 0:
                # print("PF First initialization.") # Optional print
                self.pf_core.initialize_particles(observer_location, range_measurement)
                # Store initial state for visualization
                self.current_particles_state = self.pf_core.particles_state.copy()
            else:
                # Cannot initialize without a valid first measurement
                return # Skip update cycle until initialized

        # Store previous particle state for visualization/debugging
        self.previous_particles_state = self.pf_core.particles_state.copy()

        # 2. Predict Step
        self.pf_core.predict(dt)

        # 3. Update Step (Weighting and Resampling) - Optional
        if perform_update_step:
            effective_range = range_measurement if has_new_range else -1.0

            # 3a. Update Weights
            self.pf_core.update_weights(effective_range, observer_location)

            # 3b. Resample
            self.pf_core.resample_particles(method=self.resampling_method)

            # 3c. Evaluate Filter Quality (optional, can trigger re-initialization)
            # Use a reasonable default for max error if not specified elsewhere
            # Example: 10% of max particle range for range method
            max_eval_error = 0.1 * self.pf_core.max_particle_range
            self.pf_core.evaluate_filter_quality(observer_location, effective_range, max_mean_range_error=max_eval_error)

        # 4. Estimate Target State (always do this after predict/update)
        self.pf_core.estimate_target_state(method=2) # Use weighted average

        # 5. Update public attributes
        self.estimated_location = self.pf_core.estimated_location
        self.estimated_velocity = self.pf_core.estimated_velocity
        self.current_particles_state = self.pf_core.particles_state.copy()


    def update_least_squares(self, dt: float, has_new_range: bool, range_measurement: float, observer_location: Location, num_points_to_use: int = 30):
        """
        Updates the target estimate using a simple Least Squares batch method.
        Note: This is kept separate from the PF logic.

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
        if n_points < 3: # Need at least 3 points for 2D + range offset LS
            # Cannot compute yet, maybe return previous estimate or None?
            # For consistency, let's store a placeholder if history exists
            if self.ls_state_history:
                 last_state = self.ls_state_history[-1].copy()
                 # Update position based on last velocity (simple prediction)
                 if dt > 1e-9:
                     last_state[0] += last_state[1] * dt
                     last_state[2] += last_state[3] * dt
                 self.ls_state_history.append(last_state)
            return False # Indicate LS update didn't run successfully

        # Use only the last 'num_points_to_use' points
        start_idx = max(0, n_points - num_points_to_use)
        used_x = self.ls_observer_x_history[start_idx:]
        used_y = self.ls_observer_y_history[start_idx:]
        used_z = self.ls_range_history[start_idx:]
        num = len(used_x)

        if num < 3: return False # Still not enough points after slicing

        # --- Standard Unconstrained Least Squares (LS-U) formulation ---
        # Matrix form: A * [xt, yt, -k]^T = b
        # A = [2*xi, 2*yi, -1] (for each point i)
        # b = [xi^2 + yi^2 - zi^2] (for each point i)
        # We solve for [xt, yt, -k] = pinv(A) * b

        P = np.array([used_x, used_y]) # Observer positions (2 x num)
        A = np.zeros((num, 3))
        b = np.zeros((num, 1))

        A[:, 0] = 2 * P[0, :] # 2*xi column
        A[:, 1] = 2 * P[1, :] # 2*yi column
        A[:, 2] = -1         # -1 column (for k)

        b[:, 0] = P[0, :]**2 + P[1, :]**2 - np.array(used_z)**2 # xi^2 + yi^2 - zi^2

        try:
            # Solve using pseudo-inverse (more robust than direct inversion)
            # solution = [xt, yt, -k]
            solution = np.linalg.pinv(A) @ b
            self.ls_estimated_position = solution[0:2, 0] # Extract [xt, yt]
        except np.linalg.LinAlgError:
            print('WARNING: LS solution failed (singular matrix). Skipping LS update.')
            # Keep previous estimate if available
            if self.ls_state_history:
                 last_state = self.ls_state_history[-1].copy()
                 if dt > 1e-9:
                     last_state[0] += last_state[1] * dt
                     last_state[2] += last_state[3] * dt
                 self.ls_state_history.append(last_state)
            return False

        # --- Estimate Velocity and Orientation (Simple finite difference) ---
        ls_velocity = np.array([0.0, 0.0])
        ls_orientation = 0.0

        if self.ls_state_history:
            prev_state = self.ls_state_history[-1]
            dx = self.ls_estimated_position[0] - prev_state[0]
            dy = self.ls_estimated_position[1] - prev_state[2]
            if dt > 1e-6:
                ls_velocity[0] = dx / dt # vx
                ls_velocity[1] = dy / dt # vy
            # Avoid calculating orientation from noise if change is tiny
            if np.sqrt(dx**2 + dy**2) > 1e-6:
                 ls_orientation = np.arctan2(dy, dx)
            else:
                ls_orientation = prev_state[4] # Keep previous orientation

        # Store the full state [x, vx, y, vy, orientation]
        current_ls_state = np.array([
            self.ls_estimated_position[0], ls_velocity[0],
            self.ls_estimated_position[1], ls_velocity[1],
            ls_orientation
        ])
        self.ls_state_history.append(current_ls_state)
        return True # Indicate successful LS update
