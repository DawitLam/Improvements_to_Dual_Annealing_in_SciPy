import numpy as np
from scipy.optimize import minimize

__all__ = ['memetic_climbing']

def hollow_distribution(dimension):
    # Generate initial random values for each call
    initial_randoms = np.random.uniform(-1, 1, size=dimension)
    adjusted_numbers = np.zeros(dimension)  # Initialize an array of zeros of the correct size

    for i in range(len(initial_randoms)):
        num = initial_randoms[i]
        if -0.5 < num < 0:
            adjusted_numbers[i] = num - 0.5
        elif 0 <= num < 0.5:
            adjusted_numbers[i] = num + 0.5
        else:
            adjusted_numbers[i] = num  # Keep the number as it is if it's already in the desired range

    return adjusted_numbers

def hill_climbing_with_restarts(objective, bounds, nx, cycles=1000, max_iterations=20000, step_size=20, decay_rate=0.99977, local_search=True):

    best_global_value = float('inf')
    iteration_details_global = []
    max_iterations = max_iterations // cycles

    lower_bounds, upper_bounds = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    for cycle in range(cycles):

        x0 = np.random.uniform(lower_bounds, upper_bounds, size=nx)
        best_x = np.array(x0)

        best_value = objective(best_x)

        iterations = 0

        while iterations < max_iterations:
            next_x = best_x + hollow_distribution(nx) * step_size
            next_x = np.clip(next_x, a_min=lower_bounds, a_max=upper_bounds)
            next_value = objective(next_x)

            if next_value <= best_value:
                best_x, best_value = next_x, next_value

            iterations += 1
            step_size *= decay_rate

        if local_search:
            result_bfgs = minimize(objective, best_x, bounds=bounds, method='L-BFGS-B')
            best_x_bfgs, best_value_bfgs = result_bfgs.x, result_bfgs.fun

            if best_value_bfgs <= best_value:
                best_x, best_value = best_x_bfgs, best_value_bfgs

        if best_value < best_global_value:
            best_global_value = best_value

    return iteration_details_global, {
        # 'best_x': best_global_x,
        'best_value': best_global_value,
    }

def memetic_climbing(objective, bounds, nx, cycles=1000, local_search=True):
    """Memetic algorithm that incorporates local search with hill climbing."""
    return hill_climbing_with_restarts(objective, bounds, nx, cycles=cycles)
