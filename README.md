# Improvements to Dual Annealing in SciPy

# Introduction

This project focuses on enhancing the dual annealing optimization algorithm implemented in the SciPy library. The algorithm resembels of a memetic climbing algorithm, and implemented in Python is designed for optimization tasks and incorporates local search with hill climbing.

## Project Structure

- `main.py`: This is the main script to run the improved dual annealing algorithm.
- `MemeticClimbing.py`: This script contains the implementation of the Memetic Climbing technique, which is used to enhance the performance of the dual annealing algorithm.
- `CEC2022.py`: This file includes benchmarks from the CEC2022 competition. Note that this file is not authored by us and is included for benchmarking purposes only https://github.com/P-N-Suganthan/2022-SO-BO.
- `Test.py`: A test file where users can modify it for experimenting with other evaluation objectives.


## How to Use the Algorithm

### Prerequisites

Ensure you have Python and SciPy installed.

## How to Use the use the Algorithm
1. Import the memetic_climbing function from the Memetic_Climbing.py file into your Python script:
    from Memetic_Climbing import memetic_climbing
2. Define your objective function to be optimized. Ensure that it takes a numpy array as input and returns a scalar value.

3. Define the bounds for the optimization problem as a list of tuples, where each tuple represents the lower and upper bounds for each dimension.

4. Call the memetic_climbing function with your objective function, bounds, and any additional parameters you want to specify.

5. Optionally, you can modify the Test.py file to experiment with different evaluation objectives. Ensure you understand the purpose and functionality of each file before making modifications.
