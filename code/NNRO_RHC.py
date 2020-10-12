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


RANDOM_STATE = 21

# load the dataset
def load_dataset(full_path):
    wine = pd.read_csv(full_path, delimiter=";")

    wine['quality'] = wine['quality'].replace([3,4,5,6,7,8,9],['0','0','0','1','1','1','1'])
    X = wine.iloc[:, :-1].values
    y = wine.iloc[:, -1].values

    X = preprocessing.scale(X)

    y = LabelEncoder().fit_transform(y)
    #print(y)

    return X, y

def main():
    # Load the Wine dataset
    full_path = 'data/winequality-white.csv'
    X, y = load_dataset(full_path)

    #Split data into training and test sets
    X_train_scaled, X_test_scaled, y_train_hot, y_test_hot = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Initialize neural network object and fit object
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 3000, \
                                     bias = True, is_classifier = True, learning_rate = 0.5, \
                                     early_stopping = True, clip_max = 5, restarts=2, max_attempts = 100, \
                                     random_state = RANDOM_STATE)


    nn_model1.fit(X_train_scaled, y_train_hot)

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print("Train accuracy:", y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print("Test accuracy:", y_test_accuracy)

if __name__ == "__main__":
    main()
