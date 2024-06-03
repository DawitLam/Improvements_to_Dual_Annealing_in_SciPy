import numpy as np

from Memetic_Climbing import memetic_climbing

# Define the objective function
def objective_function(x):
    return np.sum(x**2)

# Set bounds for the variables
bounds = [(-10, 10)] * 10

# Run the memetic climbing algorithm
iteration_details, result = memetic_climbing(objective_function, bounds, nx=10, cycles=1000, local_search=True)
print("Best value found:", result['best_value'])
