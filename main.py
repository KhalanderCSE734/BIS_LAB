import numpy as np

# --- Problem Definition for a Simple Two-Member Truss ---
# Objective: Minimize total weight (proportional to total area for a fixed length structure).
# Variables: A1, A2 (cross-sectional areas of the two members).
# Constraints: Stress in each member must be within allowable limits for a given load P.
# Assume a simple scenario where weight is minimized subject to stress constraints.

# Material properties and geometry
E = 200e9 # Elastic Modulus (Pa)
rho = 7850 # Density (kg/m^3)
L = 1.0 # Length of members (m)
P = 10e3 # Applied Load (N)
allowable_stress = 150e6 # Allowable stress (Pa) 
num_members = 2

# The 'fitness' function needs to handle the structural analysis and constraints.
# For simplicity, we define a hypothetical fitness that penalizes stress violations.
def calculate_stress_and_weight(areas):
    """
    Hypothetical calculation for stress and weight for a simple truss.
    In a real application, this function would call a Finite Element Analysis (FEA) tool.
    For this example, we assume stress is inversely proportional to area and proportional to load.
    A simple stress model: stress = Load * L / (Area * E)
    """
    A1, A2 = areas
    if A1 <= 0 or A2 <= 0:
        return 1e9, 1e9 # Penalize zero or negative areas
    stress1 = P * L / (A1 * E)
    stress2 = P * L / (A2 * E) # Simplified stress calculation

    # Weight is proportional to the sum of areas (assuming constant length and density)
    weight = rho * (A1 + A2) * L

    # Check constraints: stress should be within allowable limits
    # In a real GA, we combine objective and constraints into fitness.
    # We want to minimize weight, so fitness should be higher for better (lower weight, feasible) solutions.
    # Fitness = 1 / (Weight * Penalty) if invalid, else 1/Weight

    penalty = 1
    if abs(stress1) > allowable_stress or abs(stress2) > allowable_stress:
        penalty = 1000 # Large penalty for violating stress limit

    # We maximize fitness, so we return the inverse of the cost (weight * penalty)
    return 1.0 / (weight * penalty), weight, abs(stress1), abs(stress2)


def genetic_algorithm_optimization(generations, pop_size, mutation_rate):
    low_bound, high_bound = 1e-5, 1e-3
    population = np.random.uniform(low_bound, high_bound, (pop_size, num_members))

    print("Gen |    Fitness    |   A1 (m²)   |   A2 (m²)   | Weight (kg) | Stress1 (MPa) | Stress2 (MPa)")
    print("-"*95)

    for generation in range(generations):

        # 1. Fitness evaluation
        fitness_scores = []
        for sol in population:
            fitness, _, _, _ = calculate_stress_and_weight(sol)
            fitness_scores.append(fitness)
        fitness_scores = np.array(fitness_scores)

        # Best solution
        best_idx = np.argmax(fitness_scores)
        best_sol = population[best_idx]
        best_fitness, best_weight, s1, s2 = calculate_stress_and_weight(best_sol)

        # ---- LOGGING ----
        print(f"{generation+1:3d} | {best_fitness:1.4e} | "
              f"{best_sol[0]:1.4e} | {best_sol[1]:1.4e} | "
              f"{best_weight:10.4f} | {s1/1e6:10.3f} | {s2/1e6:10.3f}")

        # 2. Selection (Tournament)
        parents = []
        for _ in range(pop_size):
            indices = np.random.randint(0, pop_size, 3)
            winner = indices[np.argmax(fitness_scores[indices])]
            parents.append(population[winner])
        parents = np.array(parents)

        # 3. Crossover
        offspring = np.empty_like(population)
        for i in range(0, pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            cp = np.random.randint(1, num_members)

            offspring[i, :cp] = parent1[:cp]
            offspring[i, cp:] = parent2[cp:]
            offspring[i+1, :cp] = parent2[:cp]
            offspring[i+1, cp:] = parent1[cp:]

        # 4. Mutation
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                mutation = np.random.uniform(-0.01*high_bound,
                                             0.01*high_bound,
                                             num_members)
                offspring[i] += mutation
                offspring[i] = np.clip(offspring[i], low_bound, high_bound)

        population = offspring

    print("\n--- Optimization Complete ---")
    print(f"Optimal Areas: A1={best_sol[0]:.4e} m^2, A2={best_sol[1]:.4e} m^2")
    print(f"Minimum Weight: {best_weight:.2f} kg")
    print(f"Stress 1: {s1/1e6:.2f} MPa")
    print(f"Stress 2: {s2/1e6:.2f} MPa")

# Run the optimization
genetic_algorithm_optimization(generations=100, pop_size=50, mutation_rate=0.1)


