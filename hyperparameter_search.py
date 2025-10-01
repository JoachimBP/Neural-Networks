
# ========================================================================================== #
#                                                                                            #
#                      <<< Hyperparameter Search for Neural Network >>>                      #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#  This script performs a grid search for hyperparameters to find the optimal settings       #
#  for training the ReLU neural network. It iterates through combinations of learning        #
#  rates, momentums, and batch sizes, records the final validation loss for each run,        #
#  and reports the best-performing configurations.                                           #
#                                                                                            #
# ========================================================================================== #


#_____________________________________________________________________________________________/ Imports

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import toml
import os
import pandas as pd
import copy
from datetime import datetime

#_____________________________________________________________________________________________/ Neural Network Class

class ReLU_network:
    """
    A class to define, train, and evaluate a ReLU neural network.
    """
    def __init__(self, network_cfg, scheduler_cfg=None):
        """Initializes a neural network with a given architecture and learning rate."""
        layers = network_cfg['layers']
        learning_rate = network_cfg['learning_rate']

        # --- Bias Initializer Setup ---
        bias_mean = network_cfg.get('bias_initializer_mean', 0.0)
        bias_stddev = network_cfg.get('bias_initializer_stddev', 0.0)

        if bias_stddev > 0:
            bias_initializer = tf.keras.initializers.RandomNormal(mean=bias_mean, stddev=bias_stddev)
        else:
            bias_initializer = 'zeros'

        self.nnet = Sequential()
        self.nnet.add(Input(shape=(layers[0],)))
        for size in layers[1:-1]:
            self.nnet.add(Dense(size, activation='relu', bias_initializer=bias_initializer))
        self.nnet.add(Dense(layers[-1], activation='linear', bias_initializer=bias_initializer))
        
        # Set up the learning rate (no scheduler for this search)
        lr_schedule = learning_rate
        
        optimizer_name = network_cfg.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr_schedule)
        elif optimizer_name == 'sgd':
            momentum = network_cfg.get('momentum', 0.0)
            optimizer = SGD(learning_rate=lr_schedule, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Please use 'adam' or 'sgd'.")
        self.nnet.compile(optimizer=optimizer, loss='mse')

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size, verbose=0):
        """Trains the neural network."""
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=verbose)

#_____________________________________________________________________________________________/ Main Experiment Logic

def run_hyperparameter_search(config):
    """
    Loads data, runs hyperparameter grid search, and reports results.
    """
    # --- 1. Define Hyperparameter Grid ---
    learning_rates = [0.001, 0.002, 0.005]
    momentums = [0.0, 0.9]
    batch_sizes = [8, 16, 50]
    bias_initializer_stddevs = [0.0, 1.0, 3.0]  # Not used in this search but could be added
    num_repeats = 20  # Number of times to repeat training for each hyperparameter set

    print("Starting hyperparameter search with the following grid:")
    print(f"  Learning Rates: {learning_rates}")
    print(f"  Momentums: {momentums}")
    print(f"  Batch Sizes: {batch_sizes}")
    print(f"  Bias Initializer Stddevs: {bias_initializer_stddevs}")
    print(f"  Repetitions per set: {num_repeats}\n")

    # --- 2. Load Base Parameters & Create Target Function ---
    target_func_cfg = config['target_function']
    dataset_cfg = config['dataset']
    
    breakpoints_x = np.array(target_func_cfg['x_coords'])
    breakpoints_y = np.array(target_func_cfg['y_coords'])
    sort_indices = np.argsort(breakpoints_x)
    interpolated_function = interp1d(
        breakpoints_x[sort_indices], 
        breakpoints_y[sort_indices], 
        kind='linear', 
        fill_value="extrapolate"
    )

    # --- 3. Generate a consistent Dataset for all runs ---
    x_data = np.random.uniform(target_func_cfg['x_min'], target_func_cfg['x_max'], size=dataset_cfg['num_samples'])
    y_data = interpolated_function(x_data) + np.random.normal(0, dataset_cfg['noise_std_dev'], size=dataset_cfg['num_samples'])
    x_train = x_data.reshape(-1, 1)
    y_train = y_data.reshape(-1, 1)
    x_val = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 100).reshape(-1, 1)
    y_val = interpolated_function(x_val).reshape(-1, 1)
    print(f"✓ Dataset generated with {dataset_cfg['num_samples']} samples.")

    # --- 4. Run Grid Search ---
    results = []
    total_runs = len(learning_rates) * len(momentums) * len(batch_sizes) * len(bias_initializer_stddevs) * num_repeats
    current_run = 0

    for lr in learning_rates:
        for mom in momentums:
            for bs in batch_sizes:
                for bias_std in bias_initializer_stddevs:
                    run_losses = []
                    for i in range(num_repeats):
                        current_run += 1
                        print(f"▶ Running ({current_run}/{total_runs}): lr={lr}, momentum={mom}, batch_size={bs}, bias_stddev={bias_std}, repeat={i+1}/{num_repeats}")

                        # Create a fresh config for this run
                        run_config = copy.deepcopy(config)
                        run_config['network']['learning_rate'] = lr
                        run_config['network']['momentum'] = mom
                        run_config['network']['batch_size'] = bs
                        run_config['network']['bias_initializer_stddev'] = bias_std

                        # Initialize and train the network
                        nnet = ReLU_network(run_config['network'])
                        history = nnet.train(x_train, y_train, x_val, y_val, 
                                            epochs=run_config['network']['epochs'], 
                                            batch_size=run_config['network']['batch_size'],
                                            verbose=0)
                        
                        final_loss = history.history['val_loss'][-1]
                        run_losses.append(final_loss)

                    # Record results for this hyperparameter set
                    results.append({
                        'learning_rate': lr,
                        'momentum': mom,
                        'batch_size': bs,
                        'bias_initializer_stddev': bias_std,
                        'mean_loss': np.mean(run_losses),
                        'std_loss': np.std(run_losses),
                        'min_loss': np.min(run_losses),
                        'max_loss': np.max(run_losses)
                    })

    # --- 5. Report Results ---
    print("\n\n--- Hyperparameter Search Results ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='mean_loss', ascending=True)

    # Save results to a CSV file
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = f"hyperparameter_search_results_{date_time_str}.csv"
    results_df.to_csv(results_filename, index=False)
    
    print(f"✓ Full results saved to '{results_filename}'")
    print("\nTop 5 Best Performing Hyperparameter Sets (by mean validation loss):")
    print(results_df.head(5).to_string())

#_____________________________________________________________________________________________/ Main execution block

def main():
    """Main function to load config and run the search."""
    config_path = 'config.toml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        run_hyperparameter_search(config)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()
