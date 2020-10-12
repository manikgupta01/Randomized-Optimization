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

algorithms = ['RHC', 'SA', 'GA', 'MIMIC']

# weights = [10, 5, 2, 8, 15]
# values = [1, 2, 3, 4, 5]
# max_weight_pct = 0.6

knapsack_len=50
weights=np.random.uniform(10,40,knapsack_len)
values=np.random.uniform(20,30,knapsack_len)
max_weight_pct = 0.6


fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length = len(weights), fitness_fn = fitness, maximize = True)

# Hyperparameters - Random hill climbing
restart_arr = np.arange(0,10)
RHC_fitness_curve = np.zeros(10)
for i in range(10):
    #print(i)
    best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts=10,
            max_iters=1000, restarts=i, init_state=None, curve=False, random_state=RANDOM_STATE)
    RHC_fitness_curve[i] = best_fitness

plt.figure()
plt.plot(restart_arr, RHC_fitness_curve)
plt.title("RHC - with different restarts", fontsize=16, fontweight='bold')
plt.xlabel("Restart")
plt.ylabel("Fitness")
plt.savefig('knapsack_hp_RHC.png', bbox_inches = "tight")

# Hyperparameters - Simulated Annealing Exp Decay
temp_arr = np.arange(1.0, 100.0, 10.0)
SA_fitness_curve_exp = np.zeros(10)

for i in range(10):
    temp = (i*10) + 1
    schedule = mlrose.ExpDecay(init_temp=temp, exp_const=0.005, min_temp=0.001)
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10,
                max_iters=1000,init_state=None,curve=False,random_state=RANDOM_STATE)
    SA_fitness_curve_exp[i] = best_fitness

# Hyperparameters - Simulated Annealing Geom Decay
temp_arr = np.arange(1.0, 100.0, 10.0)
SA_fitness_curve_geom = np.zeros(10)

for i in range(10):
    temp = (i*10) + 1
    schedule = mlrose.GeomDecay(init_temp=temp, decay=0.99, min_temp=0.001)
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10,
                max_iters=1000,init_state=None,curve=False,random_state=RANDOM_STATE)
    SA_fitness_curve_geom[i] = best_fitness

plt.figure()
x = temp_arr
y_1 = SA_fitness_curve_exp
y_2 = SA_fitness_curve_geom
plt.plot(x, y_1, label="SA ExpDecay")
plt.plot(x, y_2, label="SA GeomDecay")
plt.legend(loc="best")
plt.title("SA - Decay schedule", fontsize=16, fontweight='bold')
plt.xlabel("Temperature")
plt.ylabel("Fitness")
plt.savefig('knapsack_hp_SA.png', bbox_inches = "tight")

# Hyperparameters - genetic_alg pop_size
pop_size = 200
mp = 0.1
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve1 = best_fitness_curve

pop_size = 200
mp = 0.4
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve2 = best_fitness_curve

pop_size = 200
mp = 0.9
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve3 = best_fitness_curve

pop_size = 500
mp = 0.1
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve4 = best_fitness_curve

pop_size = 500
mp = 0.4
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve5 = best_fitness_curve

pop_size = 500
mp = 0.9
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve6 = best_fitness_curve

pop_size = 800
mp = 0.1
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve7 = best_fitness_curve

pop_size = 800
mp = 0.4
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve8 = best_fitness_curve

pop_size = 800
mp = 0.9
best_state, best_fitness, best_fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mp, max_attempts=10,
            max_iters=1000, curve=True, random_state=RANDOM_STATE)
GA_fitness_curve9 = best_fitness_curve


plt.figure()
temp = max(len(GA_fitness_curve1), len(GA_fitness_curve2), len(GA_fitness_curve3), len(GA_fitness_curve4),
    len(GA_fitness_curve5), len(GA_fitness_curve6), len(GA_fitness_curve7), len(GA_fitness_curve8),
    len(GA_fitness_curve9))
x_1 = np.linspace(1, temp, len(GA_fitness_curve1))
x_2 = np.linspace(1, temp, len(GA_fitness_curve2))
x_3 = np.linspace(1, temp, len(GA_fitness_curve3))
x_4 = np.linspace(1, temp, len(GA_fitness_curve4))
x_5 = np.linspace(1, temp, len(GA_fitness_curve5))
x_6 = np.linspace(1, temp, len(GA_fitness_curve6))
x_7 = np.linspace(1, temp, len(GA_fitness_curve7))
x_8 = np.linspace(1, temp, len(GA_fitness_curve8))
x_9 = np.linspace(1, temp, len(GA_fitness_curve9))
y_1 = GA_fitness_curve1
y_2 = GA_fitness_curve2
y_3 = GA_fitness_curve3
y_4 = GA_fitness_curve4
y_5 = GA_fitness_curve5
y_6 = GA_fitness_curve6
y_7 = GA_fitness_curve7
y_8 = GA_fitness_curve8
y_9 = GA_fitness_curve9
plt.plot(x_1, y_1, label="popsize 200, mutation prob 0.1")
plt.plot(x_2, y_2, label="popsize 200, mutation prob 0.4")
plt.plot(x_3, y_3, label="popsize 200, mutation prob 0.9")
plt.plot(x_4, y_4, label="popsize 500, mutation prob 0.1")
plt.plot(x_5, y_5, label="popsize 500, mutation prob 0.4")
plt.plot(x_6, y_6, label="popsize 500, mutation prob 0.9")
plt.plot(x_7, y_7, label="popsize 800, mutation prob 0.1")
plt.plot(x_8, y_8, label="popsize 800, mutation prob 0.4")
plt.plot(x_9, y_9, label="popsize 800, mutation prob 0.9")
plt.legend(loc="best")
plt.title("GA - with popsize and mutation", fontsize=16, fontweight='bold')
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig('knapsack_hp_GA.png', bbox_inches = "tight")


# Hyperparameters - MIMIC pop_size and pct
pop_size = 200
pct = 0.2
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve1 = best_fitness_curve

pop_size = 200
pct = 0.5
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve2 = best_fitness_curve

pop_size = 200
pct = 0.9
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve3 = best_fitness_curve

pop_size = 500
pct = 0.2
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve4 = best_fitness_curve

pop_size = 500
pct = 0.5
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve5 = best_fitness_curve

pop_size = 500
pct = 0.9
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve6 = best_fitness_curve

pop_size = 800
pct = 0.2
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve7 = best_fitness_curve

pop_size = 800
pct = 0.5
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve8 = best_fitness_curve

pop_size = 800
pct = 0.9
best_state, best_fitness, best_fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=pct, max_attempts=10,
            max_iters=1000, curve=True, fast_mimic=True, random_state=RANDOM_STATE)
MIMIC_fitness_curve9 = best_fitness_curve

plt.figure()
temp = max(len(MIMIC_fitness_curve1), len(MIMIC_fitness_curve2), len(MIMIC_fitness_curve3), len(MIMIC_fitness_curve4),
    len(MIMIC_fitness_curve5), len(MIMIC_fitness_curve6), len(MIMIC_fitness_curve7), len(MIMIC_fitness_curve8),
    len(MIMIC_fitness_curve9))
x_1 = np.linspace(1, temp, len(MIMIC_fitness_curve1))
x_2 = np.linspace(1, temp, len(MIMIC_fitness_curve2))
x_3 = np.linspace(1, temp, len(MIMIC_fitness_curve3))
x_4 = np.linspace(1, temp, len(MIMIC_fitness_curve4))
x_5 = np.linspace(1, temp, len(MIMIC_fitness_curve5))
x_6 = np.linspace(1, temp, len(MIMIC_fitness_curve6))
x_7 = np.linspace(1, temp, len(MIMIC_fitness_curve7))
x_8 = np.linspace(1, temp, len(MIMIC_fitness_curve8))
x_9 = np.linspace(1, temp, len(MIMIC_fitness_curve9))
y_1 = MIMIC_fitness_curve1
y_2 = MIMIC_fitness_curve2
y_3 = MIMIC_fitness_curve3
y_4 = MIMIC_fitness_curve4
y_5 = MIMIC_fitness_curve5
y_6 = MIMIC_fitness_curve6
y_7 = MIMIC_fitness_curve7
y_8 = MIMIC_fitness_curve8
y_9 = MIMIC_fitness_curve9
plt.plot(x_1, y_1, label="popsize 200, keep_pct 0.2")
plt.plot(x_2, y_2, label="popsize 200, keep_pct 0.5")
plt.plot(x_3, y_3, label="popsize 200, keep_pct 0.9")
plt.plot(x_4, y_4, label="popsize 500, keep_pct 0.2")
plt.plot(x_5, y_5, label="popsize 500, keep_pct 0.5")
plt.plot(x_6, y_6, label="popsize 500, keep_pct 0.9")
plt.plot(x_7, y_7, label="popsize 800, keep_pct 0.2")
plt.plot(x_8, y_8, label="popsize 800, keep_pct 0.5")
plt.plot(x_9, y_9, label="popsize 800, keep_pct 0.9")
plt.legend(loc="best")
plt.title("MIMIC - with popsize and keep_pct", fontsize=16, fontweight='bold')
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig('knapsack_hp_MIMIC.png', bbox_inches = "tight")
