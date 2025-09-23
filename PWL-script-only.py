# ========================================================================================== #
#                                                                                            #
#                           <<< Neural Network Batch Training >>>                            #
#                                 (Configuration-based)                                      #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#  This script trains a ReLU neural network multiple times on a single dataset.              #
#  All parameters are loaded from 'config.toml'. It saves final losses and activation        #
#  region counts without generating any plots.                                               #
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
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_val, Y_val), verbose=0)

    def get_activation_region_count(self, x_space):
        """Finds the number of distinct linear regions over the input space."""
        x_tensor = tf.convert_to_tensor(x_space, dtype=tf.float32)
        
        binary_activations = []
        current_output = x_tensor
        for layer in self.nnet.layers:
            current_output = layer(current_output)
            if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
                binary_activations.append(tf.cast(current_output > 0, tf.int8))
        
        if not binary_activations:
            return 1 # A network with no ReLUs has only one affine region
            
        # Create a unique "signature" for each input point based on its activation pattern
        signatures = tf.concat([tf.reshape(act, (act.shape[0], -1)) for act in binary_activations], axis=1).numpy()
        
        # Count the number of changes in activation patterns
        change_indices = np.any(np.diff(signatures, axis=0), axis=1)
        
        # Number of regions = number of changes + 1
        return np.sum(change_indices) + 1

#_____________________________________________________________________________________________/ Main Experiment Logic

def run_experiment(config):
    """
    Loads data, runs multiple training loops, and saves the results.
    """
    # --- 1. Load Parameters from Config ---
    exp_cfg = config['experiment']
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

    # --- 3. Generate a Single, Shared Dataset ---
    x_data = np.random.uniform(target_func_cfg['x_min'], target_func_cfg['x_max'], size=dataset_cfg['num_samples'])
    y_data = interpolated_function(x_data) + np.random.normal(0, dataset_cfg['noise_std_dev'], size=dataset_cfg['num_samples'])
    
    x_train = x_data.reshape(-1, 1)
    y_train = y_data.reshape(-1, 1)
    
    # Use a dense grid as a consistent validation set
    x_val = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 100).reshape(-1, 1)
    y_val = interpolated_function(x_val).reshape(-1, 1)
    
    print(f"✓ Dataset generated with {dataset_cfg['num_samples']} samples.")

    # --- 4. Run Multiple Training Sessions ---
    final_losses = []
    num_activation_regions = []
    num_seen_activation_regions = []
    num_runs = exp_cfg['num_runs']
    
    # A high-resolution space to check for activation changes
    x_pred_space = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 1000).reshape(-1, 1)

    for i in range(num_runs):
        print(f"\n--- Running training {i + 1}/{num_runs} ---")
        
        # Initialize a new network for each run to get new random weights
        nnet = ReLU_network(layers=network_cfg['layers'], learning_rate=network_cfg['learning_rate'])
        
        history = nnet.train(x_train, y_train, x_val, y_val, epochs=network_cfg['epochs'])
        
        # Record final validation loss
        final_loss = history.history['val_loss'][-1]
        final_losses.append(final_loss)
        print(f"  Final Validation Loss: {final_loss:.6f}")
        
        # Record number of activation regions
        regions = nnet.get_activation_region_count(x_pred_space)
        num_activation_regions.append(regions)
        print(f"  Activation Regions: {regions}")

        sorted_indices = np.argsort(x_train.flatten())
        sorted_x_train = x_train[sorted_indices]
        seen_regions = nnet.get_activation_region_count(sorted_x_train)
        num_seen_activation_regions.append(seen_regions)
        print(f"  Seen Activation Regions: {seen_regions}")

    print("\n✓ All training runs complete.")

    # --- 5. Save Results ---
    if output_cfg.get('save_experiment', False):
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_cfg['results_base_dir'], date_time_str)
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save a copy of the config file for reproducibility
        shutil.copy('config.toml', os.path.join(output_path, 'config.toml'))
        
        # Save the results arrays to a compressed NumPy file
        results_file_path = os.path.join(output_path, 'results.npz')
        np.savez_compressed(
            results_file_path,
            final_losses=np.array(final_losses),
            activation_regions=np.array(num_activation_regions),
            seen_activation_regions=np.array(num_seen_activation_regions)
        )
        
        print(f"✓ Results saved to '{output_path}'.")

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
        run_experiment(config)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()