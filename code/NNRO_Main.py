from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
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

# load the dataset
def load_dataset(full_path):
    wine = pd.read_csv(full_path, delimiter=";")

    wine['quality'] = wine['quality'].replace([3,4,5,6,7,8,9],['0','0','0','1','1','1','1'])
    X = wine.iloc[:, :-1].values
    y = wine.iloc[:, -1].values
    #print(X.shape[1])

    X = preprocessing.scale(X)

    y = LabelEncoder().fit_transform(y)

    return X, y

def main():
    # Load the Wine dataset
    algorithms = ['RHC', 'SA', 'GA']
    clock_time = np.zeros(3)

    full_path = 'data/winequality-white.csv'
    X, y = load_dataset(full_path)

    #Split data into training and test sets
    X_train_scaled, X_test_scaled, y_train_hot, y_test_hot = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # RHC
    t_bef = time.time()
    nn_model_RHC = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 3000, \
                                     bias = True, is_classifier = True, learning_rate = 0.5, \
                                     early_stopping = True, clip_max = 5, restarts=0, max_attempts = 100, \
                                     random_state = RANDOM_STATE, curve=True)
    nn_model_RHC.fit(X_train_scaled, y_train_hot)
    t_aft = time.time()
    clock_time[0] = t_aft -  t_bef
    RHC_fitness_curve = nn_model_RHC.fitness_curve

    # SA
    t_bef = time.time()
    nn_model_SA = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
                                     algorithm = 'simulated_annealing', max_iters = 1000, \
                                     bias = True, is_classifier = True, learning_rate = 0.5, \
                                     early_stopping = True, clip_max = 5, schedule=mlrose.GeomDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001), max_attempts = 10, \
                                     random_state = RANDOM_STATE, curve=True)


    nn_model_SA.fit(X_train_scaled, y_train_hot)
    t_aft = time.time()
    clock_time[1] = t_aft -  t_bef
    SA_fitness_curve = nn_model_SA.fitness_curve

    # GA
    t_bef = time.time()
    nn_model_GA = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'tanh', \
                                     algorithm = 'genetic_alg', max_iters = 1000, \
                                     bias = True, is_classifier = True, learning_rate = 0.0001, \
                                     early_stopping = True, clip_max = 5, pop_size=200, mutation_prob=0.1, max_attempts = 100, \
                                     random_state = RANDOM_STATE, curve=True)


    nn_model_GA.fit(X_train_scaled, y_train_hot)
    t_aft = time.time()
    clock_time[2] = t_aft -  t_bef
    GA_fitness_curve = nn_model_GA.fitness_curve

    # Clock time different algorithms
    plt.figure()
    plt.barh(algorithms, clock_time, align = 'center')
    plt.title("Randomized Optimization", fontsize=16, fontweight='bold')
    plt.suptitle("Neural Network", fontsize=10)
    plt.ylabel('Algorithm')
    plt.xlabel('Time (seconds)')
    plt.savefig('RO_NN_time.png', bbox_inches = "tight")

    # Fitness curve for different algorithms
    plt.figure()
    temp = max(len(RHC_fitness_curve), len(SA_fitness_curve), len(GA_fitness_curve))
    x_1 = np.linspace(1, temp, len(RHC_fitness_curve))
    x_2 = np.linspace(1, temp, len(SA_fitness_curve))
    x_3 = np.linspace(1, temp, len(GA_fitness_curve))
    y_1 = RHC_fitness_curve
    y_2 = SA_fitness_curve
    y_3 = GA_fitness_curve
    plt.plot(x_1, y_1, 'x-', label="RHC")
    plt.plot(x_2, y_2, '.-', label="SA")
    plt.plot(x_3, y_3, '+-', label="GA")
    plt.legend(loc="best")
    plt.title("Randomized Optimization", fontsize=16, fontweight='bold')
    plt.suptitle("Neural Networks", fontsize=10)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig('RO_NN_fitness.png', bbox_inches = "tight")

    # # #RHC
    # # RHC_loss = np.zeros(5)
    # # for i in range(5):
    # #     nn_model_RHC = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
    # #                                     algorithm = 'random_hill_climb', max_iters = 3000, \
    # #                                     bias = True, is_classifier = True, learning_rate = 0.5, \
    # #                                     early_stopping = True, clip_max = 5, restarts=i, max_attempts = 100, \
    # #                                     random_state = RANDOM_STATE, curve=True)
    # #     nn_model_RHC.fit(X_train_scaled, y_train_hot)
    # #     RHC_loss[i] = nn_model_RHC.loss
    # #
    # # #print(RHC_loss)
    # # RHC_restart = np.arange(0,5,1)
    # # # print(RHC_restart)
    # #
    # # # RHC Plot RHC curve
    # # plt.figure()
    # # plt.plot(RHC_restart, RHC_loss)
    # # plt.title("Randomized Optimization", fontsize=16, fontweight='bold')
    # # plt.suptitle("Neural Network Loss Curve RHC", fontsize=10)
    # # plt.ylabel('Loss')
    # # plt.xlabel('Restart')
    # # plt.savefig('RO_NN_loss_RHC.png', bbox_inches = "tight")
    #
    # #GA
    # GA_loss = np.zeros(3)
    # nn_model_GA = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'tanh', \
    #                                 algorithm = 'genetic_alg', max_iters = 1000, \
    #                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
    #                                 early_stopping = True, clip_max = 5, pop_size=200, mutation_prob=0.1, max_attempts = 100, \
    #                                 random_state = RANDOM_STATE, curve=True)
    # nn_model_GA.fit(X_train_scaled, y_train_hot)
    # GA_loss[0] = nn_model_GA.loss
    #
    # nn_model_GA = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'tanh', \
    #                                 algorithm = 'genetic_alg', max_iters = 1000, \
    #                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
    #                                 early_stopping = True, clip_max = 5, pop_size=500, mutation_prob=0.1, max_attempts = 100, \
    #                                 random_state = RANDOM_STATE, curve=True)
    # nn_model_GA.fit(X_train_scaled, y_train_hot)
    # GA_loss[1] = nn_model_GA.loss
    #
    # nn_model_GA = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'tanh', \
    #                                 algorithm = 'genetic_alg', max_iters = 1000, \
    #                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
    #                                 early_stopping = True, clip_max = 5, pop_size=900, mutation_prob=0.1, max_attempts = 100, \
    #                                 random_state = RANDOM_STATE, curve=True)
    # nn_model_GA.fit(X_train_scaled, y_train_hot)
    # GA_loss[2] = nn_model_GA.loss
    #
    # #print(RHC_loss)
    # GA_popsize = [200, 500, 900]
    # # print(RHC_restart)
    #
    # # RHC Plot RHC curve
    # plt.figure()
    # plt.plot(GA_popsize, GA_loss)
    # plt.title("Randomized Optimization", fontsize=16, fontweight='bold')
    # plt.suptitle("Neural Network Loss Curve GA", fontsize=10)
    # plt.ylabel('Loss')
    # plt.xlabel('Population Size')
    # plt.savefig('RO_NN_loss_GA.png', bbox_inches = "tight")
    #
    # nn_model_SA = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
    #                                  algorithm = 'simulated_annealing', max_iters = 1000, \
    #                                  bias = True, is_classifier = True, learning_rate = 0.9, \
    #                                  early_stopping = True, clip_max = 5, schedule=mlrose.ExpDecay(init_temp=20.0, exp_const=0.005, min_temp=0.001), max_attempts = 10, \
    #                                  random_state = RANDOM_STATE, curve=False)

if __name__ == "__main__":
    main()
