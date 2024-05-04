import numpy as np


class Swarm:
    def __init__(self, bounds, f, tol, n_particles=50, c1=2.05, c2=2.05):
        self.dim = len(bounds)
        self.bounds = list(zip(*bounds))  # [(x1_min,...,x_dim_min),...,(x1_max,...,x_dim_max)]
        self.f = lambda x: f(*x)  # vectorized version of the objective function
        self.tol = tol

        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        c = c1 + c2
        self.constriction_factor = 2 / abs(2 - c - np.sqrt(c ** 2 - 4 * c))

        self.rng = np.random.default_rng(seed=42)
        self.positions, self.velocities = self.rng.uniform(self.bounds[0], self.bounds[1], (2, n_particles, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.local_best_positions = np.zeros((n_particles, self.dim))
        self.update_local_bests()

        self.epoch_count = 0

    def minimize(self):
        while self.has_not_converged():
            self.update_swarm()

    def has_not_converged(self, max_n_epochs=2000):
        condition1 = np.linalg.norm(np.var(self.positions, axis=0)) > self.tol
        condition2 = self.epoch_count < max_n_epochs

        if not condition1:
            print("The swarm converged successfully")
            self.print_info()
        if not condition2:
            print("Reached maximum number of epochs")
            self.print_info()

        return condition1 and condition2

    def update_swarm(self):
        r1, r2 = self.rng.uniform(size=(2, self.n_particles, self.dim))

        self.velocities = self.constriction_factor * (
                self.velocities +
                self.c1 * r1 * (self.local_best_positions - self.positions) +
                self.c2 * r2 * (self.personal_best_positions - self.positions))
        self.positions += self.velocities

        self.keep_within_bounds()

        self.update_personal_bests()
        self.update_local_bests()

        self.epoch_count += 1

    def keep_within_bounds(self):
        for j in range(self.dim):
            jth_coords = self.positions[:, j]
            min_pos = self.bounds[0][j]
            max_pos = self.bounds[1][j]

            jth_coords[jth_coords < min_pos] = min_pos
            jth_coords[jth_coords > max_pos] = max_pos

    def update_personal_bests(self):
        for i in range(self.n_particles):
            self.personal_best_positions[i] = min(self.positions[i], self.personal_best_positions[i], key=self.f)

    def update_local_bests(self):
        left_neighbours = np.roll(self.positions, -1, axis=0)
        right_neighbours = np.roll(self.positions, 1, axis=0)

        for i in range(self.n_particles):
            self.local_best_positions[i] = min(np.vstack((left_neighbours[i],
                                                          self.positions[i],
                                                          right_neighbours[i])), key=self.f)

    def get_f_values(self):
        return [self.f(pos) for pos in self.positions]

    def get_min_point(self):
        return self.positions[np.argmin(self.get_f_values())]

    def print_info(self):
        print("Number of epochs:", self.epoch_count)
        print("Minimum: {} at {}".format(self.f(self.get_min_point()), self.get_min_point()))
