import numpy as np
import matplotlib.pyplot as plt
import random

# ==== Parameters ====
GRID_SIZE = (20, 20)
NUM_WAYPOINTS = 5
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.2

START = (0, 0)
GOAL = (19, 19)

# === Obstacles (some rectangles) ===
obstacles = [
    ((5, 5), (10, 6)),
    ((7, 12), (15, 13)),
    ((12, 2), (13, 10)),
]

# ==== Helper Functions ====

def is_collision(p1, p2):
    """Rough collision check: if a line crosses any obstacle box area."""
    x1, y1 = p1
    x2, y2 = p2
    for (ox1, oy1), (ox2, oy2) in obstacles:
        if (min(x1, x2) <= ox2 and max(x1, x2) >= ox1 and
            min(y1, y2) <= oy2 and max(y1, y2) >= oy1):
            return True
    return False

def total_path_length(path):
    return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))

def fitness(path):
    length = total_path_length(path)
    collision_penalty = 0
    for i in range(len(path)-1):
        if is_collision(path[i], path[i+1]):
            collision_penalty += 100  # big penalty
    return length + collision_penalty

def generate_random_path():
    waypoints = []
    for _ in range(NUM_WAYPOINTS):
        x = random.randint(0, GRID_SIZE[0] - 1)
        y = random.randint(0, GRID_SIZE[1] - 1)
        waypoints.append((x, y))
    return [START] + waypoints + [GOAL]

def mutate(path):
    new_path = path.copy()
    for i in range(1, len(path) - 1):
        if random.random() < MUTATION_RATE:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            x = max(0, min(GRID_SIZE[0]-1, path[i][0] + dx))
            y = max(0, min(GRID_SIZE[1]-1, path[i][1] + dy))
            new_path[i] = (x, y)
    return new_path

def crossover(p1, p2):
    cut = random.randint(1, NUM_WAYPOINTS)
    return [START] + p1[1:cut+1] + p2[cut+1:-1] + [GOAL]

# ==== Initialize Population ====
population = [generate_random_path() for _ in range(POPULATION_SIZE)]

# ==== Evolution Loop ====
print("Starting Gene Expression Algorithm for Path Planning...\n")

for generation in range(GENERATIONS):
    population = sorted(population, key=fitness)
    best_fit = fitness(population[0])

    print(f"Generation {generation+1}/{GENERATIONS} | Best Fitness: {best_fit:.2f}")

    new_population = population[:10]  # elitism: top 10 kept

    while len(new_population) < POPULATION_SIZE:
        parent1 = random.choice(population[:25])
        parent2 = random.choice(population[:25])
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

# ==== Final Result ====
best_path = min(population, key=fitness)
print("\n✅ Optimization Complete!")
print("Best Path Fitness:", fitness(best_path))
print("Best Path Coordinates:")
for point in best_path:
    print(f"→ {point}")

# ==== Visualization ====
def draw_path(path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, GRID_SIZE[0])
    ax.set_ylim(0, GRID_SIZE[1])

    # Draw obstacles
    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red')
        ax.add_patch(rect)

    # Draw path
    xs, ys = zip(*path)
    ax.plot(xs, ys, marker='o', color='blue', label='Best Path')

    # Draw start and goal
    ax.plot(START[0], START[1], 'go', markersize=10, label='Start')
    ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10, label='Goal')

    ax.grid(True)
    ax.legend()
    plt.title("Robot Path Planning via GEA")
    plt.show()

draw_path(best_path)
