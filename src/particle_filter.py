import numpy as np
import random
from world_objects import Object, Velocity, Location

SOUND_SPEED = 1500.0


class ParticleFilter:
    """Particle Filter for target tracking."""

    def __init__(self, std_range, init_velocity, state_dim,
                 num_particles=6000, method='range', max_range=250):
        self.std_range = std_range
        self.init_velocity = init_velocity
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.method = method
        self.max_range = max_range

        self.particles = np.zeros((num_particles, state_dim))
        self.old_particles = np.zeros((num_particles, state_dim))
        self.estimated_state = np.zeros(state_dim)

        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.velocity_noise = 0.0

        self.velocity = 0
        self.orientation = 0

        self.weights = np.ones(num_particles)
        self.covariance_vals = [0.02, 0.02]
        self.covariance_theta = 0.0
        self.initialized = False

        self.last_measurement = 0
        self.prev_distances = np.zeros(num_particles)
        self.old_weights = self.weights.copy()
        self.prev_observer = Location(0, 0, 0)

        self.cov_matrix = np.ones((2, 2))

    def estimate(self):
        """Estimate target state from the particles using a weighted mean."""
        weighted_sum = np.zeros(self.state_dim)
        for i in range(self.num_particles):
            weighted_sum += self.particles[i] * self.weights[i]
        self.estimated_state = weighted_sum / np.sum(self.weights)
        self.velocity = np.sqrt(
            self.estimated_state[1]**2 + self.estimated_state[3]**2)
        self.orientation = np.arctan2(
            self.estimated_state[3], self.estimated_state[1])

        x_coords = self.particles[:, 0]
        y_coords = self.particles[:, 2]
        self.cov_matrix = np.cov(x_coords, y_coords)
        return

    def initialize_particles(self, position, lateral_range):
        """Initialize particles in a circular region around the given position."""
        for i in range(self.num_particles):
            angle = 2 * np.pi * np.random.rand()
            if self.method == 'area':
                r = np.random.rand() * self.max_range * 2 - self.max_range
            else:
                r = np.random.rand() * self.std_range * 2 - self.std_range + lateral_range
            self.particles[i, 0] = r * np.cos(angle) + position.x
            self.particles[i, 2] = r * np.sin(angle) + position.y
            orientation = np.random.rand() * 2 * np.pi
            v = random.gauss(self.init_velocity, self.init_velocity / 2)
            self.particles[i, 1] = np.cos(orientation) * v
            self.particles[i, 3] = np.sin(orientation) * v
        self.estimate()
        self.initialized = True

    def set_noise(self, forward_noise, turn_noise, sense_noise, velocity_noise):
        """Set the noise parameters for motion and sensing."""
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.velocity_noise = velocity_noise

    def predict(self, dt):
        """Predict the new state of each particle over a time interval dt."""
        for i in range(self.num_particles):
            current_angle = np.arctan2(
                self.particles[i, 3], self.particles[i, 1])
            orientation = (current_angle + (np.random.rand() *
                           self.turn_noise * 2 - self.turn_noise)) % (2 * np.pi)
            velocity = np.sqrt(
                self.particles[i, 1]**2 + self.particles[i, 3]**2)
            distance = velocity * dt + \
                (np.random.rand() * self.forward_noise * 2 - self.forward_noise)
            self.particles[i, 0] += np.cos(orientation) * distance
            self.particles[i, 2] += np.sin(orientation) * distance
            new_velocity = velocity + \
                (np.random.rand() * self.velocity_noise * 2 - self.velocity_noise)
            new_velocity = max(new_velocity, 0)
            self.particles[i, 1] = np.cos(orientation) * new_velocity
            self.particles[i, 3] = np.sin(orientation) * new_velocity

    def gaussian(self, prev_dist, curr_dist, sigma, prev_meas, curr_meas, inc_observer):
        """Compute the probability using a Gaussian (or Cauchy for area method)."""
        if self.method == 'area':
            sigma = 1.0
            if curr_meas != -1:
                return 0.5 - (1 / np.pi) * np.arctan((curr_dist - self.max_range) / sigma)
            else:
                sigma = 40.0
                return 0.5 + (1 / np.pi) * np.arctan((curr_dist - self.max_range) / sigma)
        else:
            return np.exp(-((curr_dist - curr_meas) ** 2) / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

    def update_measurements(self, measurement, observer):
        """Update particle weights based on the measurement."""
        distances = []
        for i in range(self.num_particles):
            dist = np.sqrt((self.particles[i, 0] - observer.x)**2 +
                           (self.particles[i, 2] - observer.y)**2)
            prev_dist = np.sqrt((self.particles[i, 0] - self.prev_observer.x)**2 +
                                (self.particles[i, 2] - self.prev_observer.y)**2)
            inc_observer = np.sqrt((observer.x - self.prev_observer.x)**2 +
                                   (observer.y - self.prev_observer.y)**2)
            self.weights[i] = self.gaussian(prev_dist, dist, self.sense_noise,
                                            self.last_measurement, measurement, inc_observer)
            distances.append(dist)
        self.last_measurement = measurement
        self.prev_distances = np.array(distances)
        self.old_weights = self.weights.copy()
        self.prev_observer = Location(observer.x, observer.y, observer.depth)

    def resample(self, z):
        """Resample particles based on their weights using one of several methods."""
        method = 2 if (
            self.estimated_state[0] == 0 and self.estimated_state[2] == 0) else 3

        if method == 1:
            new_particles = np.zeros((self.num_particles, self.state_dim))
            index = int(np.random.random() * self.num_particles)
            beta = 0.0
            mw = max(self.weights)
            for i in range(self.num_particles):
                beta += np.random.random() * 2.0 * mw
                while beta > self.weights[index]:
                    beta -= self.weights[index]
                    index = (index + 1) % self.num_particles
                new_particles[i] = self.particles[index]
            self.particles = new_particles
        elif method == 2:
            new_particles = np.zeros((self.num_particles, self.state_dim))
            cumulative_sum = np.cumsum(self.weights / np.sum(self.weights))
            u0 = np.random.random() / self.num_particles
            i = 0
            for j in range(self.num_particles):
                while u0 > cumulative_sum[i]:
                    i += 1
                new_particles[j] = self.particles[i]
                u0 += 1.0 / self.num_particles
            self.particles = new_particles
        elif method in (3, 3.2):
            if self.num_particles == 10000:
                ratio = 640
            elif self.num_particles == 6000:
                ratio = 400
            elif self.num_particles == 3000:
                ratio = 200
            elif self.num_particles == 1000:
                ratio = 120
            else:
                ratio = 50
            radii = 0.2 if method == 3 else self.max_range
            new_particles = np.zeros((self.num_particles, self.state_dim))
            cumulative_sum = np.cumsum(self.weights / np.sum(self.weights))
            u0 = np.random.random() / (self.num_particles - ratio)
            i = 0
            for j in range(self.num_particles - ratio):
                while u0 > cumulative_sum[i]:
                    i += 1
                new_particles[j] = self.particles[i]
                u0 += 1.0 / (self.num_particles - ratio)
            for j in range(ratio):
                aux = np.zeros(4)
                angle = 2 * np.pi * np.random.rand()
                r = np.random.rand() * radii
                if method == 3:
                    aux[0] = r * np.cos(angle) + self.estimated_state[0]
                    aux[2] = r * np.sin(angle) + self.estimated_state[2]
                else:
                    aux[0] = r * np.cos(angle) + self.prev_observer.x
                    aux[2] = r * np.sin(angle) + self.prev_observer.y
                orientation = np.random.rand() * 2 * np.pi
                v = random.gauss(self.init_velocity, self.init_velocity / 2)
                aux[1] = np.cos(orientation) * v
                aux[3] = np.sin(orientation) * v
                new_particles[self.num_particles - ratio + j] = aux
                self.weights[self.num_particles - ratio +
                             j] = 1.0 / (self.num_particles / 3.0)
            self.particles = new_particles

    def evaluate(self, observer, z, max_error=50):
        """Evaluate the filter performance and reset if necessary."""
        if self.method != 'area':
            total_error = 0.0
            for i in range(self.num_particles):
                dx = self.particles[i, 0] - observer.x
                dy = self.particles[i, 2] - observer.y
                total_error += np.sqrt(dx**2 + dy**2)
            avg_error = abs(total_error / self.num_particles - z)
            err_x = self.particles[:, 0] - self.estimated_state[0]
            err_y = self.particles[:, 2] - self.estimated_state[2]
            cov = np.cov(err_x, err_y)
            vals, vecs = np.linalg.eig(cov)
            confidence_int = 2.326**2
            self.covariance_vals = np.sqrt(vals) * confidence_int
            vec_x, vec_y = vecs[:, 0]
            self.covariance_theta = np.arctan2(vec_y, vec_x)
            if avg_error > max_error and np.sqrt(self.covariance_vals[0]**2 + self.covariance_vals[1]**2) < 5.0:
                self.initialized = False
        else:
            if np.max(self.weights) < 0.1:
                self.initialized = False
        return


class TargetPF(Object):
    """Particle filter based target that is compatible with world_objects.py."""

    def __init__(self, method='range', max_range=250, dt=1.0):
        super().__init__(location=Location(0.0, 0.0, 0.0),
                         velocity=Velocity(0.0, 0.0, 0.0),
                         name="estimated_landmark")
        self.method = method
        self.pf = ParticleFilter(std_range=0.02, init_velocity=0.2, state_dim=4,
                                 num_particles=1000, method=method, max_range=max_range)
        self.pf.set_noise(forward_noise=0.01, turn_noise=0.1,
                          sense_noise=0.005, velocity_noise=0.01)
        self.pf_state = np.zeros(4)
        self.ls_history = []
        self.ls_easting = []
        self.ls_northing = []
        self.ls_solution = None
        self.all_measurements = []
        self.dt = dt

    def update_particle_filter(self, new_range, measurement, observer, update=True):
        """Update the particle filter and the target's state estimation."""
        max_error = 50
        if update:
            if not self.pf.initialized:
                self.pf.initialize_particles(
                    position=observer, lateral_range=measurement)
            self.pf.old_particles = self.pf.particles.copy()
            self.pf.predict(self.dt)
            if new_range:
                self.pf.update_measurements(measurement, observer)
                self.pf.resample(measurement)
                self.pf.evaluate(observer, measurement, max_error)
            self.pf.estimate()
        self.pf_state = self.pf.estimated_state.copy()
        # Update the location of this object instead of separate attributes
        self.location.x = self.pf_state[0]
        self.location.y = self.pf_state[2]
        # Depth can remain unchanged or be updated as needed
        self.velocity = Velocity(self.pf_state[1], self.pf_state[3], 0.0)
        return True

    def update_least_squares(self, dt, new_range, measurement, observer):
        """Update the least squares estimation (LS)."""
        num_points_used = 30
        if new_range:
            self.all_measurements.append(measurement)
            self.ls_easting.append(observer.x)
            self.ls_northing.append(observer.y)
        if len(self.ls_easting) > 3:
            P = np.matrix([self.ls_easting[-num_points_used:],
                          self.ls_northing[-num_points_used:]])
            N = np.concatenate((np.identity(2), np.zeros((2, 1))), axis=1)
            num = len(self.ls_easting[-num_points_used:])
            A = np.concatenate((2 * P.T, -np.ones((num, 1))), axis=1)
            b = np.matrix(
                (np.diag(P.T * P) - np.array(self.all_measurements[-num_points_used:])**2)).T
            try:
                self.ls_solution = N @ np.linalg.inv(A.T @ A) @ A.T @ b
            except np.linalg.LinAlgError:
                try:
                    self.ls_solution = N @ np.linalg.inv(
                        A.T @ A + 1e-6) @ A.T @ b
                except Exception:
                    print('WARNING: LS singular matrix')
            try:
                ls_orientation = np.arctan2(self.ls_solution[1] - self.ls_history[-1][2],
                                            self.ls_solution[0] - self.ls_history[-1][0])
            except IndexError:
                ls_orientation = 0
            try:
                ls_velocity = np.array([(self.ls_solution[0] - self.ls_history[-1][0]) / dt,
                                       (self.ls_solution[1] - self.ls_history[-1][1]) / dt])
            except IndexError:
                ls_velocity = np.array([0, 0])
            try:
                ls_position = np.array([self.ls_solution.item(0),
                                        ls_velocity.item(0),
                                        self.ls_solution.item(1),
                                        ls_velocity.item(1),
                                        ls_orientation if not hasattr(ls_orientation, 'item') else ls_orientation.item(0)])
            except IndexError:
                ls_position = np.array(
                    [observer.x, ls_velocity[0], observer.y, ls_velocity[1], ls_orientation])
            self.ls_history.append(ls_position)
        return True
