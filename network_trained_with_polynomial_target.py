# ========================================================================================== #
#                                                                                            #
#                <<< Réseau de neurones approximant un polynôme >>>                          #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#       Outil pour visualiser le comportement d'un réseau de neurone approximant un          #
#    polynôme en fonction du nombre d'epochs.                                                #
#                                                                                            #
# ========================================================================================== #



#_____________________________________________________________________________________________/ Importation :


import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd
import tensorflow as tf
from keras import Input
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


#_____________________________________________________________________________________________/ Polynôme approximé :

def P(x) :
    return -0.01*x**3 + 0.33*x**2 - 3*x + 8


#_____________________________________________________________________________________________/ Données d'entraînement:

def create_noisy_dataset(f, epsilon, n, a, b) :
    ''' 
        Créer des échantillons qui suivent une loi Y ~ f(X) + U([-e, e]).
        Retour sous la forme d'une liste d'inputs (x) et d'outputs (y).

    '''
    def noise() :
        return rd.uniform(-epsilon, epsilon)
    x = np.array([rd.uniform(a, b) for i in range(n)])
    y = np.array([f(i) + noise() for i in x])
    return x, y

def create_dataset(f, epsilon, n, a, b) :
    ''' 
        Créer des échantillons qui suivent une loi Y ~ f(X).
        Retour sous la forme d'une liste d'inputs (x) et d'outputs (y).

    '''
    x = np.array([rd.uniform(a, b) for i in range(n)])
    y = np.array([f(i) for i in x])
    return x, y


#_____________________________________________________________________________________________/ Réseau de neurones:

class ReLU_network :
    def __init__(self, layers) :
        ''' Initialise un réseau ReLU d'architecture (layers) '''
        self.input_size = layers[0]
        self.output_size = layers[-1]
        self.size = len(layers)

        self.nnet = Sequential()
        self.nnet.add(Input(shape=(self.input_size,)))
        for size in layers[2:-1]:
            self.nnet.add(Dense(size, activation='relu'))
        self.nnet.add(Dense(self.output_size, activation='linear'))
        optimizer = Adam(learning_rate=0.005)
        self.nnet.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train(self, x_train, y_train, epoch) :
        return self.nnet.fit(x_train, y_train, epochs=epoch, batch_size=32, verbose=0)

    def evaluate(self, X) :
        return self.nnet.predict(X)

    def __str__(self) :
        self.nnet.summary()
        return ""


#_____________________________________________________________________________________________/ Main :


def main() :
    a, b, epsilon, n = 0, 20, 1, 50
    nnet = ReLU_network([1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20 ,1])
    print(nnet)
    x = np.linspace(a, b, 100)
    y = P(x)
    plt.plot(x, y, label = "$P$")
    X = x.reshape(-1, 1)
    Y = nnet.evaluate(X)
    plt.plot(x, Y, label = "model prediction (0 epoch)")
    dataset_x, dataset_y = create_dataset(P, epsilon, n, a, b)
    plt.plot(dataset_x, dataset_y, "x", label = "training data")
    nnet.train(dataset_x, dataset_y, 10)
    X = x.reshape(-1, 1)
    Y = nnet.evaluate(X)
    plt.plot(x, Y, '--',  label = "model prediction (10 epochs)")
    nnet.train(dataset_x, dataset_y, 40)
    X = x.reshape(-1, 1)
    Y = nnet.evaluate(X)
    plt.plot(x, Y, '--',  label = "model prediction (50 epochs)")
    nnet.train(dataset_x, dataset_y, 50)
    X = x.reshape(-1, 1)
    Y = nnet.evaluate(X)
    plt.plot(x, Y, '--', label = "model prediction (100 epochs)")
    nnet.train(dataset_x, dataset_y, 100)
    X = x.reshape(-1, 1)
    Y = nnet.evaluate(X)
    plt.plot(x, Y, '--', label = "model prediction (200 epochs)")    
    plt.legend()
    plt.show()

if __name__ == "__main__" :
    main()
