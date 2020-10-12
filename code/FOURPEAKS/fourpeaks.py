import numpy as np
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

RANDOM_STATE = 21

clock_time = np.zeros(4)

algorithms = ['RHC', 'SA', 'GA', 'MIMIC']

fitness = mlrose.FourPeaks(t_pct=0.15)
init_state = np.random.randint(2, size=10)

problem = mlrose.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)

# Random hill climbing
t_bef = time.time()
best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_attempts=10,
            max_iters=1000, restarts=4, init_state=init_state, curve=True, random_state=RANDOM_STATE)
t_bef = time.time()
clock_time[0] = t_bef - t_bef
RHC_fitness_curve = fitness_curve

# print("----------------------------------")
# print("Random Hill")
# # print(best_state)
# # print(best_fitness)
# # print(fitness_curve)
# # print("----------------------------------")


# Simulated annealing
schedule = mlrose.GeomDecay(init_temp=2.0, decay=0.99, min_temp=0.001)
t_bef = time.time()
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10,
            max_iters=1000,init_state=init_state,curve=True,random_state=RANDOM_STATE)
t_aft = time.time()
clock_time[1] = t_aft - t_bef
SA_fitness_curve = fitness_curve

# print("----------------------------------")
# print("Simulated annealing")
# # print(best_state)
# # print(best_fitness)
# # print(fitness_curve)
# # print("----------------------------------")


# Genetic algorithm
t_bef = time.time()
best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=800, mutation_prob=0.1, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
t_aft = time.time()
clock_time[2] = t_aft - t_bef
GA_fitness_curve = fitness_curve

# print("----------------------------------")
# print("Genetic algorithm")
# # print(best_state)
# # print(best_fitness)
# # print(fitness_curve)
# # print("----------------------------------")


# MIMIC
t_bef = time.time()
best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=800, keep_pct=0.2, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
t_aft = time.time()
clock_time[3] = t_aft - t_bef
MIMIC_fitness_curve = fitness_curve

# print("----------------------------------")
# print("MIMIC")
# # # print(best_state)
# # # print(best_fitness)
# # # print(fitness_curve)
# # # print("----------------------------------")


# Clock time different algorithms
plt.figure()
plt.barh(algorithms, clock_time, align = 'center')
plt.title("Randomized Optimization", fontsize=16, fontweight='bold')
plt.suptitle("fourpeaks (10 samples)", fontsize=10)
plt.ylabel('Algorithm')
plt.xlabel('Time (seconds)')
plt.savefig('time_fourpeaks10.png', bbox_inches = "tight")


# Fitness curve for different algorithms
plt.figure()
temp = max(len(RHC_fitness_curve), len(SA_fitness_curve), len(GA_fitness_curve), len(MIMIC_fitness_curve))
x_1 = np.linspace(1, temp, len(RHC_fitness_curve))
x_2 = np.linspace(1, temp, len(SA_fitness_curve))
x_3 = np.linspace(1, temp, len(GA_fitness_curve))
x_4 = np.linspace(1, temp, len(MIMIC_fitness_curve))
y_1 = RHC_fitness_curve
y_2 = SA_fitness_curve
y_3 = GA_fitness_curve
y_4 = MIMIC_fitness_curve
plt.plot(x_1, y_1, 'x-', label="RHC")
plt.plot(x_2, y_2, '.-', label="SA")
plt.plot(x_3, y_3, '+-', label="GA")
plt.plot(x_4, y_4, '*-', label="MIMIC")
plt.legend(loc="best")
plt.title("Randomized Optimization", fontsize=16, fontweight='bold')
plt.suptitle("fourpeaks (10 samples)", fontsize=10)
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig('fitness_fourpeaks10.png', bbox_inches = "tight")
