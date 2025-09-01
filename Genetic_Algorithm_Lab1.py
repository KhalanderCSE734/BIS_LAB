import random

# Constants
POP_SIZE = 10               # Population size
CHROMOSOME_LENGTH = 5       # Chromosome length (0-31)
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
N_GENERATIONS = 50          # Number of generations to run

# Fitness function: f(x) = x^2
def fitness(x):
    return x ** 2

# Generate initial population (random 5-bit strings)
def generate_population():
    return [''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH)) for _ in range(POP_SIZE)]

# Convert binary to decimal
def decode(chromosome):
    return int(chromosome, 2)

# Selection: Roulette Wheel
def select_mating_pool(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:  # Avoid division by zero
        probabilities = [1 / len(population)] * len(population)
    else:
        probabilities = [f / total_fitness for f in fitness_values]
    selected = random.choices(population, weights=probabilities, k=POP_SIZE)
    return selected

# Crossover (single point)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

# Mutation (bit flip)
def mutate(chromosome):
    return ''.join(
        bit if random.random() > MUTATION_RATE else '0' if bit == '1' else '1'
        for bit in chromosome
    )

def genetic_algorithm():
    population = generate_population()
    best_solution = None
    best_fitness = float('-inf')

    for gen in range(1, N_GENERATIONS + 1):
        fitness_values = [fitness(decode(ind)) for ind in population]

        # Track best solution in this generation
        for i, fit_val in enumerate(fitness_values):
            if fit_val > best_fitness:
                best_fitness = fit_val
                best_solution = population[i]

        # Print generation stats
        print(f"Generation {gen}: Best Fitness = {best_fitness}, Best Chromosome = {best_solution}, x = {decode(best_solution)}")

        # Selection
        mating_pool = select_mating_pool(population, fitness_values)

        # Crossover
        next_generation = []
        for i in range(0, POP_SIZE, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[(i + 1) % POP_SIZE]  # wrap around
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])

        # Mutation
        population = [mutate(child) for child in next_generation[:POP_SIZE]]  # maintain pop size

    print("\n=== Final Best Solution ===")
    print(f"Chromosome: {best_solution}")
    print(f"x value: {decode(best_solution)}")
    print(f"Fitness: {best_fitness}")

genetic_algorithm()
