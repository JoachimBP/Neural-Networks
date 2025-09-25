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
# from time import time
# from sklearn.model_selection import train_test_split

# from keras.datasets import mnist

# from Fonctions.Fonctions import functional_dimension



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
    

    def get_Jacobian(self, X):
        """
        Computes the Jacobian of the network output with respect to the parameters for each input sample.

        Args:
            X (np.ndarray): Input data of shape (N_inputs, d_in).

        Returns:
            np.ndarray: An array containing the stacked Jacobians, with a final
                        shape of (N_inputs * d_out, N_par).
        """
        # Ensure the input is a TensorFlow tensor for gradient computation
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        N_inputs = X_tensor.shape[0]
        
        # Get the output dimension and total number of parameters from the model
        d_out = self.nnet.output_shape[-1]
        N_par = self.nnet.count_params()

        with tf.GradientTape(persistent=True) as tape:
            # Perform a forward pass. The tape will watch the model's trainable variables by default.
            predictions = self.nnet(X_tensor)

        # Compute the Jacobian of the model's output with respect to each trainable variable (weights and biases).
        # This returns a list of tensors, one for each layer's parameters.
        jacobians_list = tape.jacobian(predictions, self.nnet.trainable_variables)

        # The tape is persistent, so it's good practice to delete it when done.
        del tape

        # Reshape and concatenate the list of Jacobian tensors into a single matrix.
        # 1. Flatten the parameter dimensions for each tensor in the list.
        #    The shape of jacobians_list[i] is (N_inputs, d_out, *param_shape_i),
        #    which we reshape to (N_inputs, d_out, num_params_i).
        flattened_jacobians = [tf.reshape(j, (N_inputs, d_out, -1)) for j in jacobians_list]

        # 2. Concatenate the flattened Jacobians along the parameter axis (axis=2).
        #    This creates a single tensor of shape (N_inputs, d_out, N_par).
        full_jacobian = tf.concat(flattened_jacobians, axis=2)

        # 3. Reshape to the final desired format: (N_inputs * d_out, N_par).
        final_jacobian = tf.reshape(full_jacobian, (N_inputs * d_out, N_par))

        return final_jacobian.numpy()
    
    def get_rank(self, X):
        """
        Computes the rank of the Jacobian matrix for the given input data.

        Args:
            X (np.ndarray): Input data of shape (N_inputs, d_in).

        Returns:
            int: The rank of the Jacobian matrix.
        """
        J = self.get_Jacobian(X)
        rank = np.linalg.matrix_rank(J)
        return rank

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
    print("Shape of x_train:", x_train.shape)
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
    rank = nnet.get_rank(x_train)
    
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

