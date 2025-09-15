import numpy as np

# Levy flight function
def levy_flight(beta=1.5, size=1):
    sigma_u = np.power(np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / np.math.gamma((1 + beta) / 2) 
                       * beta * np.power(2, beta - 1), 1 / beta)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

# Cuckoo Search Algorithm
class CuckooSearch:
    def __init__(self, func, n_nests=25, max_iter=100, pa=0.25, beta=1.5, bounds=(-5, 5)):
        self.func = func  # Objective function
        self.n_nests = n_nests  # Number of nests
        self.max_iter = max_iter  # Maximum number of iterations
        self.pa = pa  # Probability of discovering an egg
        self.beta = beta  # Levy flight distribution parameter
        self.bounds = bounds  # Bounds for solution space
        self.n_dim = len(bounds) if isinstance(bounds, tuple) else len(bounds[0])
        
    def initialize_nests(self):
        nests = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_nests, self.n_dim))
        return nests
    
    def fitness(self, solution):
        return self.func(solution)  # Evaluate the fitness of a solution

    def generate_new_solution(self, nest, best_nest):
        levy = levy_flight(self.beta, self.n_dim)
        new_solution = nest + levy * (nest - best_nest)  # Levy flight update rule
        # Boundaries check
        new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
        return new_solution
    
    def abandon_worst_nests(self, nests, fitness_vals):
        worst_idx = np.argsort(fitness_vals)[-1]  # Index of the worst nest
        abandon_prob = np.random.rand(self.n_nests)
        abandon_idx = abandon_prob < self.pa
        abandon_idx[worst_idx] = True  # Always abandon the worst nest

        # Replace worst nests with new random solutions
        nests[abandon_idx] = np.random.uniform(self.bounds[0], self.bounds[1], 
                                                (np.sum(abandon_idx), self.n_dim))
        return nests

    def optimize(self):
        # Initialize nests and fitness
        nests = self.initialize_nests()
        fitness_vals = np.apply_along_axis(self.fitness, 1, nests)
        
        # Track the best solution
        best_idx = np.argmin(fitness_vals)
        best_solution = nests[best_idx]
        best_fitness = fitness_vals[best_idx]
        
        # Main loop (iterations)
        for t in range(self.max_iter):
            # Generate new solutions for each nest
            for i in range(self.n_nests):
                new_solution = self.generate_new_solution(nests[i], best_solution)
                new_fitness = self.fitness(new_solution)
                
                # If new solution is better, update the nest
                if new_fitness < fitness_vals[i]:
                    nests[i] = new_solution
                    fitness_vals[i] = new_fitness
                    
            # Update the best solution found so far
            best_idx = np.argmin(fitness_vals)
            best_solution = nests[best_idx]
            best_fitness = fitness_vals[best_idx]
            
            # Abandon the worst nests
            nests = self.abandon_worst_nests(nests, fitness_vals)
            
            # Print iteration progress (Optional)
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {best_fitness}")

        return best_solution, best_fitness

# Define the objective function (example: f(x) = x^2)
def objective_function(x):
    return np.sum(x**2)

# Example usage:
cuckoo = CuckooSearch(func=objective_function, n_nests=25, max_iter=100, pa=0.25, beta=1.5, bounds=(-5, 5))
best_solution, best_fitness = cuckoo.optimize()

print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
