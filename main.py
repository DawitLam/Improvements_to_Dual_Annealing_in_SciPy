
import numpy as np
import pandas as pd
import time
from CEC2022 import cec2022_func
from Memetic_Climbing import memetic_climbing

# Set dimensions and bounds for the problem
nx = 10
mx = 10

bounds = [(-100.0, 100.0)] * nx
number_trials = 30
best_known_values = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]

# Initialize CEC object
CEC = cec2022_func(func_num=0)

# Initialize results storage
HC_results = {fx_n: [] for fx_n in range(1, 13)}

start_time = time.time()

# Loop over all 12 functions
for fx_n in range(1, 13):
    CEC.func = fx_n
    function_results = []

    # Run optimization process number_trials times for the current function
    for t in range(number_trials):
        np.random.seed(t)

        def objective(x):
            return CEC.values(np.array(x).reshape(nx, 1)).ObjFunc[0]

        # Run memetic climbing and record best solution
        _, HC_Solution = memetic_climbing(objective, bounds, nx, local_search=True)
        function_results.append(HC_Solution['best_value'])

    # Record results for this function
    HC_results[fx_n] = function_results
    print(f"Function {fx_n}: Best Value among Trials = {min(function_results):.4f}")

function_time = time.time() - start_time
print(f"\nTotal time taken for the whole algorithm = {function_time:.4f} seconds")

# Prepare data for the DataFrame
data_HC = [[fx_n] + [result - best_known_values[fx_n - 1] for result in HC_results[fx_n]] for fx_n in range(1, 13)]
columns = ["Function"] + [f"Trial_{i + 1}" for i in range(number_trials)]
df_HC = pd.DataFrame(data_HC, columns=columns)

# Calculate mean and standard deviation for each function's results
df_HC["Mean"] = df_HC.iloc[:, 1:].mean(axis=1)
df_HC["Std_Dev"] = df_HC.iloc[:, 1:].std(axis=1)

# Save DataFrame to a CSV file
df_name = 'HC_MC_1000.csv'
df_HC.to_csv(df_name, index=False)


# if __name__ == '__main__':
