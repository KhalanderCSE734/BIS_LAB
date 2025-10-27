import numpy as np

# Objective function: maximize total power output from solar and wind
# Example: power = 10*x1 + 15*x2 - penalty for oversize
def objective(x):
    x1, x2 = x[0], x[1]
    # Constraint: total cost should not exceed budget (e.g. 100 units)
    cost = 5*x1 + 7*x2
    budget = 100
    penalty = 0
    if cost > budget:
        penalty = 1000 * (cost - budget)  # heavy penalty if cost exceeds budget

    # Total power output (simplified)
    power = 10 * x1 + 15 * x2
    return -(power - penalty)  # negative because GWO minimizes by default

# Grey Wolf Optimizer Implementation
class GreyWolfOptimizer:
    def __init__(self, obj_func, lb, ub, dim, n_wolves=5, max_iter=50):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.n_wolves = n_wolves
        self.max_iter = max_iter

        # Initialize the positions of search agents
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

    def optimize(self):
        for l in range(self.max_iter):
            for i in range(self.n_wolves):
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness = self.obj_func(self.positions[i])

                # Update Alpha, Beta, Delta
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            a = 2 - l * (2 / self.max_iter)  # a decreases linearly from 2 to 0

            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3

            print(f"Iteration {l+1}: Best fitness = {-self.alpha_score:.4f}, Best position = {self.alpha_pos}")

        return self.alpha_pos, -self.alpha_score

# Define problem bounds for solar panels and wind turbines sizes
lb = [0, 0]  # Minimum size
ub = [20, 15]  # Maximum size (arbitrary units)

gwo = GreyWolfOptimizer(objective, lb, ub, dim=2, n_wolves=10, max_iter=50)
best_position, best_power = gwo.optimize()

print(f"\nOptimal sizes: Solar panels = {best_position[0]:.2f}, Wind turbines = {best_position[1]:.2f}")
print(f"Maximum power output = {best_power:.2f} units")
