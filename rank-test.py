# ========================================================================================== #
#                                                                                            #
#                         <<< Single Neural Network Training & Plot >>>                      #
#                                   (Configuration-based)                                    #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#  This script trains a single ReLU neural network using parameters from 'config.toml'.      #
#  It then generates and saves a plot comparing the target function against the final        #
#  trained network's prediction.                                                             #
#                                                                                            #
# ========================================================================================== #


#_____________________________________________________________________________________________/ Imports

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam
import toml
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt




# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import SGD
# import numpy as np
from time import time
from sklearn.model_selection import train_test_split

# from keras.datasets import mnist

from Fonctions.Fonctions import functional_dimension



#_____________________________________________________________________________________________/ Neural Network Class

class ReLU_network:
    """
    A class to define, train, and evaluate a ReLU neural network.
    """
    def __init__(self, layers, learning_rate=0.01):
        """Initializes a neural network with a given architecture and learning rate."""
        self.nnet = Sequential()
        self.nnet.add(Input(shape=(layers[0],)))
        for size in layers[1:-1]:
            self.nnet.add(Dense(size, activation='relu'))
        self.nnet.add(Dense(layers[-1], activation='linear'))
        
        optimizer = Adam(learning_rate=learning_rate)
        self.nnet.compile(optimizer=optimizer, loss='mse')

    def train(self, X_train, Y_train, X_val, Y_val, epochs):
        """Trains the neural network."""
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_val, Y_val), verbose=1)

    def evaluate(self, X):
        """Predicts outputs for given inputs."""
        return self.nnet.predict(X, verbose=0)

#_____________________________________________________________________________________________/ Main Experiment Logic

def run_single_training_and_plot(config):
    """
    Loads data, trains a single network, and saves a comparison plot.
    """
    # --- 1. Load Parameters from Config ---
    target_func_cfg = config['target_function']
    dataset_cfg = config['dataset']
    network_cfg = config['network']
    output_cfg = config['output']
    print("✓ Configuration loaded.")

    # --- 2. Create Target Function ---
    breakpoints_x = np.array(target_func_cfg['x_coords'])
    breakpoints_y = np.array(target_func_cfg['y_coords'])
    
    if len(breakpoints_x) < 2:
        raise ValueError("Target function requires at least 2 breakpoints.")
    
    sort_indices = np.argsort(breakpoints_x)
    interpolated_function = interp1d(
        breakpoints_x[sort_indices], 
        breakpoints_y[sort_indices], 
        kind='linear', 
        fill_value="extrapolate"
    )
    print("✓ Target function created.")

    # --- 3. Generate Dataset ---
    x_data = np.random.uniform(target_func_cfg['x_min'], target_func_cfg['x_max'], size=dataset_cfg['num_samples'])
    y_data = interpolated_function(x_data) + np.random.normal(0, dataset_cfg['noise_std_dev'], size=dataset_cfg['num_samples'])
    
    x_train = x_data.reshape(-1, 1)
    y_train = y_data.reshape(-1, 1)
    
    x_val = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 100).reshape(-1, 1)
    y_val = interpolated_function(x_val).reshape(-1, 1)
    
    print(f"✓ Dataset generated with {dataset_cfg['num_samples']} samples.")

    # --- 4. Initialize and Train the Neural Network ---
    print(f"\n▶ Training network for {network_cfg['epochs']} epochs...")
    nnet = ReLU_network(layers=network_cfg['layers'], learning_rate=network_cfg['learning_rate'])
    history = nnet.train(x_train, y_train, x_val, y_val, epochs=network_cfg['epochs'])
    print("✓ Training complete.")
    
    final_loss = history.history['val_loss'][-1]
    print(f"  Final Validation Loss: {final_loss:.6f}")

    # --- 5. Generate Predictions for Plotting ---
    x_pred_space = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 1000).reshape(-1, 1)
    y_pred = nnet.evaluate(x_pred_space)

    # --- 6. Compute ranks for Plotting ---
    fdim = functional_dimension(nnet)

    ## X_train
    fdim.get_differential(x_train)
    rank = fdim.compute_rank()
    print(f"  Rank on training data: {rank}")

    # --- 7. Plotting Results ---
    plt.figure(figsize=(10, 7))
    
    # Plot target function
    plt.plot(x_pred_space, interpolated_function(x_pred_space), 'b-', label='Target Function', linewidth=2)
    plt.scatter(breakpoints_x[sort_indices], breakpoints_y[sort_indices], color='blue', zorder=5, label='Breakpoints')
    
    # Plot network prediction
    plt.plot(x_pred_space, y_pred, 'g-', label='NN Prediction', linewidth=2)
    
    # Plot training samples
    plt.scatter(x_train, y_train, color='orange', alpha=0.5, s=20, label='Training Samples')
    
    plt.title('Target Function vs. Trained Network Prediction', fontsize=16)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output (y)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0) # Adjust as needed

    # --- 8. Save and Finalize ---
    if output_cfg.get('save_experiment', False):
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_cfg['results_base_dir'], f"single_run_{date_time_str}")
        
        os.makedirs(output_path, exist_ok=True)
        
        shutil.copy('config.toml', os.path.join(output_path, 'config.toml'))
        
        plot_filename = os.path.join(output_path, 'prediction_vs_target.png')
        plt.savefig(plot_filename)
        
        print(f"\n✓ Plot saved to '{plot_filename}'.")
    
    plt.close()

#_____________________________________________________________________________________________/ Main execution block

def main():
    """Main function to load config and run the experiment."""
    config_path = 'config.toml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        run_single_training_and_plot(config)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()









from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
from time import time
from sklearn.model_selection import train_test_split

from keras.datasets import mnist

from Fonctions.Fonctions import functional_dimension



import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Download MNIST data
(X_1, y_1), (X_2, y_2) = mnist.load_data()
X_1 = X_1.reshape(60000, 784)
X_2 = X_2.reshape(10000, 784)
X_1 = X_1.astype('float32')
X_2 = X_2.astype('float32')
X_1 /= 255
X_2 /= 255

# Mix train and test data
X = np.concatenate((X_1, X_2), axis =0)
y = np.concatenate((y_1, y_2))

# Choose the desired sizes for train and test
train_size = 6000
test_size = 20000

# Randomly split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = train_size)
X_test = X_test[0:test_size,:]
y_test = y_test[0:test_size]

# Randomly generate gaussian inputs
random_size = test_size
X_random = np.random.normal(random_size*784*[0.5],1).reshape(random_size, 784)




    ##



w_list = []

train_spec = []
test_spec = []
random_spec = []

accuracy = []
val_accuracy = []
loss = []
val_loss = []
computation_time = []
epoch = []



t0 = time()

# Creation of the network
w = 30   # Width of the network

N0 = 784
N1 = w
N2 = w
N3 = w
NS = 10

Arch = [N0, N1, N2, N3, NS]
L = len(Arch) - 1

model = Sequential()
model.add(Dense(N1,  input_dim=N0))
model.add(Activation('ReLU'))
model.add(Dense(N2))
model.add(Activation('ReLU'))
model.add(Dense(N3))
model.add(Activation('ReLU'))
model.add(Dense(NS))
model.add(Activation('softmax'))

# Number of parameters and number of identifiable parameters (i.e. maximum theoretical functional dimension)
Npar = model.count_params()
Npar_identif = Npar - np.sum(Arch) + Arch[0] + Arch[L]

# Parameters of the training
from keras.optimizers import SGD
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# Convert class vectors to binary class matrices
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

batch_size = 256 

# Epoch increment, i.e. number of SGD epochs between two computations
# This number increases during training
nb_epoch_1 = 40
nb_epoch_2 = 200
nb_epoch_3 = 400
nb_iter_1 = 10
nb_iter_2 = 5
nb_iter_3 = 4
nb_iter = nb_iter_1 + nb_iter_2 +  nb_iter_3
total_epoch = nb_epoch_1 * nb_iter_1 + nb_epoch_2 * nb_iter_2 + nb_epoch_3 * nb_iter_3


nb_epoch = nb_epoch_1   # Current epoch increment

for cpt in range(nb_iter):
    eprint('Iteration', cpt)

    # After a certain number of iterations, change the epoch increment
    if cpt == nb_iter_1:   
        nb_epoch = nb_epoch_2
    if cpt == nb_iter_1 + nb_iter_2:
        nb_epoch = nb_epoch_3
    t0 = time()

    # Train the network
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,verbose=0,validation_data = (X_test, Y_test))

    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    eprint('acc:', acc, '- val_acc:', val_acc, '- loss:', loss, '- val_loss:', val_loss)

    # Store the current total number of epochs
    if cpt <  nb_iter_1:
        epoch.append((cpt + 1) * nb_epoch_1)
    elif cpt < nb_iter_1 + nb_iter_2:
        epoch.append(nb_iter_1 * nb_epoch_1 + (cpt + 1 - nb_iter_1) * nb_epoch_2)
    else:
        epoch.append(nb_iter_1 * nb_epoch_1 + nb_iter_2 * nb_epoch_2 + (cpt + 1 - nb_iter_1 - nb_iter_2) * nb_epoch_3)

    # Store the accuracies and losses
    accuracy.append(acc)
    val_accuracy.append(val_acc)
    loss.append(loss)
    val_loss.append(val_loss)



  
    # Compute the batch functional dimension for the trained model and several input choices
    fdim = functional_dimension(model)

    ## X_train
    fdim.get_differential(X_train)
    spec = fdim.compute_svd()
    train_spec.append(spec)

    ## X_test
    fdim.get_differential(X_test)
    spec = fdim.compute_svd()
    test_spec.append(spec)

    ## X_random
    fdim.get_differential(X_random)
    spec = fdim.compute_svd()
    random_spec.append(spec)

    
    ##

    t1 = time()

    comp_time = round((t1 - t0)/60, 2)

    eprint('Computation time:', comp_time)
    computation_time.append(comp_time)


    ##


# Store the lists of computed spectra in separate files for further analysis
import pickle
pickle.dump(train_spec, open('train_spec.dat', 'wb'))
pickle.dump(test_spec, open('test_spec.dat', 'wb'))
pickle.dump(random_spec, open('random_spec.dat', 'wb'))



print('Batch size:', batch_size, ' - Total number of epochs:', total_epoch)
print('Architecture =', Arch)
print('Number of parameters:', Npar)
print('Number of identifiable parameters:', Npar_identif)

print('Train size:', train_size, '- Test_size:', test_size, '- Random size:', random_size)
print('Epochs =', epoch)
print('Final_train_loss =', loss)
print('Final_test_loss =', val_loss)
print('Final_train_accuracy =', accuracy)
print('Final_test_accuracy =', val_accuracy)
print('Computation_time =', computation_time)