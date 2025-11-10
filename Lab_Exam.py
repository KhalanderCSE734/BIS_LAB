import math
import random
import statistics
import time
from typing import List, Tuple

import sys


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def greedy_route_length(start: Tuple[float, float], tasks: List[Tuple[float, float]]) -> float:
    if not tasks:
        return 0.0
    unvisited = tasks.copy()
    cur = start
    dist = 0.0
    while unvisited:
        idx, best = min(enumerate(unvisited), key=lambda iv: euclidean(cur, iv[1]))
        dist += euclidean(cur, best)
        cur = best
        unvisited.pop(idx)
    return dist


def decode_assignment(x: List[float], n_robots: int) -> List[int]:
    return [int(round(val)) % n_robots for val in x]


def fitness_assignment(
    x: List[float],
    robot_starts: List[Tuple[float, float]],
    task_positions: List[Tuple[float, float]],
    w_makespan: float = 0.7,
    w_total: float = 0.2,
    w_imbalance: float = 0.1,
) -> float:
    """Compute fitness: lower is better.

    - makespan: max distance any robot travels
    - total: sum of all distances
    - imbalance: stddev of loads (distances)
    """
    n_robots = len(robot_starts)
    assign = decode_assignment(x, n_robots)
    robots_tasks = [[] for _ in range(n_robots)]
    for t_idx, r in enumerate(assign):
        robots_tasks[r].append(task_positions[t_idx])

    robot_dists = []
    for r_idx in range(n_robots):
        d = greedy_route_length(robot_starts[r_idx], robots_tasks[r_idx])
        robot_dists.append(d)

    makespan = max(robot_dists) if robot_dists else 0.0
    total = sum(robot_dists)
    imbalance = statistics.pstdev(robot_dists) if len(robot_dists) > 1 else 0.0

    fitness = w_makespan * makespan + w_total * total + w_imbalance * imbalance
    return fitness


def levy_flight(dim: int, levy_lambda: float = 1.5) -> List[float]:
    sigma_u = (
        (math.gamma(1 + levy_lambda) * math.sin(math.pi * levy_lambda / 2))
        / (math.gamma((1 + levy_lambda) / 2) * levy_lambda * 2 ** ((levy_lambda - 1) / 2))
    ) ** (1 / levy_lambda)
    sigma_v = 1.0
    step = []
    for _ in range(dim):
        u = random.gauss(0, sigma_u)
        v = random.gauss(0, sigma_v)
        s = u / (abs(v) ** (1 / levy_lambda))
        step.append(s)
    return step


def cuckoo_search(
    n_robots: int,
    task_positions: List[Tuple[float, float]],
    robot_starts: List[Tuple[float, float]],
    n_nests: int = 25,
    pa: float = 0.25,
    step_size: float = 0.01,
    levy_lambda: float = 1.5,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[List[int], float, dict]:
    num_tasks = len(task_positions)

    population = [[random.uniform(0, n_robots - 1) for _ in range(num_tasks)] for _ in range(n_nests)]
    fitness_values = [fitness_assignment(sol, robot_starts, task_positions) for sol in population]

    best_index = min(range(n_nests), key=lambda i: fitness_values[i])
    best_solution_vector = population[best_index][:]
    best_fitness = fitness_values[best_index]

    history_best = [best_fitness]
    stagnation_count = 0

    for iteration in range(1, max_iter + 1):
        for i in range(n_nests):
            levy_step = levy_flight(num_tasks, levy_lambda)
            candidate = [population[i][d] + step_size * levy_step[d] for d in range(num_tasks)]
            candidate = [max(0.0, min(val, n_robots - 1)) for val in candidate]
            candidate_fitness = fitness_assignment(candidate, robot_starts, task_positions)

            j = random.randrange(n_nests)
            if candidate_fitness < fitness_values[j]:
                population[j] = candidate
                fitness_values[j] = candidate_fitness

        n_abandon = int(pa * n_nests)
        if n_abandon > 0:
            worst_indices = sorted(range(n_nests), key=lambda i: fitness_values[i], reverse=True)[:n_abandon]
            for idx in worst_indices:
                population[idx] = [random.uniform(0, n_robots - 1) for _ in range(num_tasks)]
                fitness_values[idx] = fitness_assignment(population[idx], robot_starts, task_positions)

        current_best_index = min(range(n_nests), key=lambda i: fitness_values[i])
        current_best_fitness = fitness_values[current_best_index]
        if current_best_fitness < best_fitness - tol:
            best_fitness = current_best_fitness
            best_solution_vector = population[current_best_index][:]
            stagnation_count = 0
        else:
            stagnation_count += 1

        history_best.append(best_fitness)

        if verbose:
            decoded_assignment = decode_assignment(best_solution_vector, n_robots)
            print(
                f"Iter {iteration}/{max_iter} | best_fitness: {best_fitness:.6f} | best_assignment: {decoded_assignment}"
            )

        if stagnation_count >= 50:
            if verbose:
                print(f"Converged by stagnation at iter {iteration}")
            break

    best_assignment = decode_assignment(best_solution_vector, n_robots)
    stats = {"history": history_best, "best_vector": best_solution_vector}
    return best_assignment, best_fitness, stats


def sample_scenario():
    """Create a sample scenario with 4 robots and 12 tasks."""
    random.seed(42)
    n_robots = 4
    robot_starts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    task_positions = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(12)]
    return n_robots, robot_starts, task_positions


def main():
    n_robots, robot_starts, task_positions = sample_scenario()

    n_nests = 30
    pa = 0.25 
    step_size = 0.1 
    levy_lambda = 1.5  
    max_iter = 500

    print("Running Cuckoo Search for multi-robot task allocation")
    print(f"Robots: {n_robots}, Tasks: {len(task_positions)}, Nests: {n_nests}")

    start_time = time.time()
    best_assign, best_fit, stats = cuckoo_search(
        n_robots,
        task_positions,
        robot_starts,
        n_nests=n_nests,
        pa=pa,
        step_size=step_size,
        levy_lambda=levy_lambda,
        max_iter=max_iter,
        verbose=True,
    )
    elapsed = time.time() - start_time

    print("\nResult")
    print(f"Best fitness: {best_fit:.6f}")
    print(f"Assignment (task -> robot): {best_assign}")

    robots_tasks = [[] for _ in range(n_robots)]
    for t_idx, r in enumerate(best_assign):
        robots_tasks[r].append(task_positions[t_idx])

    robot_dists = [greedy_route_length(robot_starts[i], robots_tasks[i]) for i in range(n_robots)]
    for i, d in enumerate(robot_dists):
        print(f" Robot {i}: {len(robots_tasks[i])} tasks, route length {d:.3f}")

    print(f"Elapsed: {elapsed:.2f}s, iterations: {len(stats['history'])}")


if __name__ == "__main__":
    main()
