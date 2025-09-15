import numpy as np
import random

# Step 1: Define the cities and their coordinates
city_coords = np.array([
    [0, 0], [1, 5], [5, 2], [3, 8], [7, 7],
    [9, 0], [6, 5], [4, 6], [8, 3], [2, 1]
])

num_cities = len(city_coords)
distance_matrix = np.zeros((num_cities, num_cities))

# Calculate the distance matrix
for i in range(num_cities):
    for j in range(num_cities):
        distance_matrix[i][j] = np.linalg.norm(city_coords[i] - city_coords[j])

# Parameters
num_ants = 10
num_iterations = 10
alpha = 1.0       # pheromone importance
beta = 5.0        # distance importance
rho = 0.5         # pheromone evaporation rate
Q = 100           # total pheromone deposited
initial_pheromone = 1.0

# Initialize pheromone matrix
pheromone = np.full((num_cities, num_cities), initial_pheromone)

# Heuristic information
heuristic = 1 / (distance_matrix + np.eye(num_cities))  # avoid division by zero

# Store best solution
best_path = None
best_distance = np.inf

# Main ACO loop
for iteration in range(num_iterations):
    all_paths = []
    iteration_best_path = None
    iteration_best_distance = np.inf
    
    for ant in range(num_ants):
        path = []
        visited = set()
        current_city = random.randint(0, num_cities - 1)
        path.append(current_city)
        visited.add(current_city)

        while len(visited) < num_cities:
            probabilities = []
            for next_city in range(num_cities):
                if next_city not in visited:
                    tau = pheromone[current_city][next_city] ** alpha
                    eta = heuristic[current_city][next_city] ** beta
                    probabilities.append((next_city, tau * eta))
            total = sum(prob for _, prob in probabilities)
            if total == 0:
                next_city = random.choice([c for c in range(num_cities) if c not in visited])
            else:
                r = random.uniform(0, total)
                s = 0
                for city, prob in probabilities:
                    s += prob
                    if s >= r:
                        next_city = city
                        break

            path.append(next_city)
            visited.add(next_city)
            current_city = next_city

        path.append(path[0])  # Return to start city
        distance = sum(distance_matrix[path[i]][path[i+1]] for i in range(num_cities))
        all_paths.append((path, distance))

        # Track best path for this ant
        if distance < iteration_best_distance:
            iteration_best_path = path
            iteration_best_distance = distance

    # Update global best solution if the iteration best is better
    if iteration_best_distance < best_distance:
        best_path = iteration_best_path
        best_distance = iteration_best_distance

    # Pheromone evaporation
    pheromone *= (1 - rho)

    # Pheromone update
    for path, dist in all_paths:
        for i in range(num_cities):
            from_city = path[i]
            to_city = path[i+1]
            pheromone[from_city][to_city] += Q / dist
            pheromone[to_city][from_city] += Q / dist

    # Print best move in this iteration
    print(f"Iteration {iteration + 1}: Best Path: {iteration_best_path} with Distance: {iteration_best_distance:.2f}")

# Final result
print("\nFinal Best Path:", best_path)
print(f"Final Best Distance: {best_distance:.2f}")
