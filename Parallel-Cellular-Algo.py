import numpy as np


np.random.seed(0)
costs = np.random.randint(1, 10, (5, 5))


distances = np.full((5, 5), np.inf)
distances[0, 0] = 0  # starting point

def get_neighbors(i, j, shape):
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
            neighbors.append((ni, nj))
    return neighbors

def update_distances(distances, costs):
    new_distances = distances.copy()
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            neighbors = get_neighbors(i, j, costs.shape)
            neighbor_values = [distances[x, y] + costs[i, j] for x, y in neighbors]
            best_val = min([distances[i, j]] + neighbor_values)
            new_distances[i, j] = best_val
    return new_distances

for step in range(10):
    new_distances = update_distances(distances, costs)
    if np.allclose(new_distances, distances):
        print(f"Converged after {step+1} iterations.")
        break
    distances = new_distances

print("Cost Grid (each cell's movement cost):")
print(costs)
print("\nShortest Distance to Goal (Parallel Cellular Computation):")
print(np.round(distances, 2))
print(f"\nEstimated shortest distance from start to goal: {distances[-1, -1]:.2f}")
