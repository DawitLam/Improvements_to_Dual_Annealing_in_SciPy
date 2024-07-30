"""
Memetic Climbing Algorithm
==========================

This module provides an implementation of the memetic climbing algorithm, which combines hill climbing with 
restarts and local search using L-BFGS-B.

Functions
---------
memetic_climbing(objective, bounds, nx, cycles=1000, local_search=True)
    Memetic algorithm that incorporates local search with hill climbing.

hollow_distribution(dimension)
    Generate initial random values for each call.

hill_climbing_with_restarts(objective, bounds, nx, cycles=1000, max_iterations=20000, step_size=20, decay_rate=0.99977, local_search=True)
    Hill climbing optimization with restarts.

Authors
-------
Dawit Gulta (dawit.lambebo@gmail.com)
Stephen Chen (sychen@yorku.ca)
Copyright (c) 2024 Dawit Gulta (dawit.lambebo@gmail.com), Stephen Chen (sychen@yorku.ca)
"""

import numpy as np
from scipy.optimize import minimize

__all__ = ['memetic_climbing']

def hollow_distribution(dimension):
    """
    Generate initial random values for each call
    Parameters
    ----------
    dimension : int
        The dimensionality of the distribution.

    Returns
    -------
    adjusted_numbers : ndarray
        An array of adjusted random values.
    """
    
    
    initial_randoms = np.random.uniform(-1, 1, size=dimension)
    adjusted_numbers = np.zeros(dimension)  # Initialize an array of zeros of the correct size
    """
        To enhance exploration compared to a simple uniform random distribution,
        the 'next step' was made not just ±10 from the current location.
        Instead, it is a uniform value from the range [5, 10],
        applied in either the positive or negative direction.
    """

    
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

     """Hill climbing optimization with restarts.

    Parameters
    ----------
    objective : callable
        The objective function to be minimized.
    bounds : sequence
        Bounds for variables (min, max).
    nx : int
        Number of dimensions.
    cycles : int, optional
        Number of cycles to perform (default is 1000).
    max_iterations : int, optional
        Maximum number of iterations (default is 20000).
    step_size : float, optional
        Step size for hill climbing (default is 20).
    decay_rate : float, optional
        Decay rate for step size (default is 0.99977).
    local_search : bool, optional
        Whether to perform local search using L-BFGS-B (default is True).

    Returns
    -------
    iteration_details_global : list
        List of details for each iteration.
    best_result : dict
        Dictionary containing the best found value.
    """
    
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
    """Memetic algorithm that incorporates local search with hill climbing.

    Parameters
    ----------
    objective : callable
        The objective function to be minimized.
    bounds : sequence
        Bounds for variables (min, max).
    nx : int
        Number of dimensions.
    cycles : int, optional
        Number of cycles to perform (default is 1000).
    local_search : bool, optional
        Whether to perform local search using L-BFGS-B (default is True).

    Returns
    -------
    best_result : dict
        Dictionary containing the best found value.
    """
    return hill_climbing_with_restarts(objective, bounds, nx, cycles=cycles, local_search=local_search)

"""
import numpy as np
from scipy.optimize import memetic_climbing

def test_memetic_climbing():
    """Test the memetic climbing optimization algorithm."""
    # Define the objective function
    def objective_function(x):
        return np.sum(x**2)

    # Set bounds for the variables
    bounds = [(-10, 10)] * 10

    # Run the memetic climbing algorithm
    iteration_details, result = memetic_climbing(objective_function, bounds, nx=10, cycles=1000, local_search=True)

    # Assert the results are within the expected range
    assert result['best_value'] < 1e-8, "Optimization did not converge to the expected minimum."

"""
