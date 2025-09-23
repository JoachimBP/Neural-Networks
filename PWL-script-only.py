# ========================================================================================== #
#                                                                                            #
#                             <<< Neural Network Training >>>                                #
#                               (Configuration-based)                                        #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#  This script trains a ReLU neural network to approximate a piecewise-affine function.      #
#  All parameters are loaded from a 'config.toml' file, removing the need for a GUI.         #
#                                                                                            #
# ========================================================================================== #


#_____________________________________________________________________________________________/ Imports

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam
import toml  # Library to parse .toml files
import os

#_____________________________________________________________________________________________/ Neural Network Class (largely unchanged)

class ReLU_network:
    """
    A class to define, train, and evaluate a ReLU neural network.
    """
    def __init__(self, layers, learning_rate=0.01):
        """Initializes a neural network with a given architecture and learning rate."""
        self.input_size = layers[0]
        self.output_size = layers[-1]
        self.size = len(layers)

        self.nnet = Sequential()
        self.nnet.add(Input(shape=(self.input_size,)))
        for size in layers[1:-1]:
            self.nnet.add(Dense(size, activation='relu'))
        self.nnet.add(Dense(self.output_size, activation='linear'))
        
        optimizer = Adam(learning_rate=learning_rate)
        self.nnet.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train(self, X_train, Y_train, X_test, Y_test, epochs):
        """Trains the neural network."""
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

    def evaluate(self, X):
        """Predicts outputs for given inputs."""
        return self.nnet.predict(X, verbose=0)

    def get_activation_change_points(self, x):
        """Finds the x-values where the activation pattern of ReLU neurons changes."""
        current_output = tf.convert_to_tensor(x, dtype=tf.float32)
        binary_activations = []
        for layer in self.nnet.layers:
            current_output = layer(current_output)
            if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
                binary_activations.append(tf.cast(current_output > 0, tf.int8))
        
        if not binary_activations:
            return np.array([])
            
        signatures = tf.concat([tf.reshape(act, (act.shape[0], -1)) for act in binary_activations], axis=1).numpy()
        change_indices = np.any(np.diff(signatures, axis=0), axis=1)
        return x[1:][change_indices].flatten()

#_____________________________________________________________________________________________/ Main Experiment Logic

def run_experiment(config):
    """
    Loads data, trains the network, and generates plots based on the provided configuration.
    """
    # --- 1. Load Parameters from Config ---
    target_func_cfg = config['target_function']
    dataset_cfg = config['dataset']
    network_cfg = config['network']
    plot_cfg = config['plot']
    output_cfg = config['output']
    print("✓ Configuration loaded.")

    # --- 2. Create Target Function ---
    breakpoints_x = np.array(target_func_cfg['x_coords'])
    breakpoints_y = np.array(target_func_cfg['y_coords'])
    
    if len(breakpoints_x) < 2:
        raise ValueError("Target function requires at least 2 breakpoints.")
    
    # Sort points by x-coordinate to ensure correct interpolation
    sort_indices = np.argsort(breakpoints_x)
    breakpoints_x = breakpoints_x[sort_indices]
    breakpoints_y = breakpoints_y[sort_indices]
    
    interpolated_function = interp1d(breakpoints_x, breakpoints_y, kind='linear', fill_value="extrapolate")
    print("✓ Target function created.")

    # --- 3. Generate Dataset ---
    x_data = np.random.uniform(plot_cfg['x_min'], plot_cfg['x_max'], size=dataset_cfg['num_samples'])
    y_data = interpolated_function(x_data) + np.random.normal(0, dataset_cfg['noise_std_dev'], size=dataset_cfg['num_samples'])
    
    x_train = x_data.reshape(-1, 1)
    y_train = y_data.reshape(-1, 1)
    print(f"✓ Dataset generated with {dataset_cfg['num_samples']} samples.")

    # --- 4. Initialize and Train the Neural Network ---
    nnet = ReLU_network(layers=network_cfg['layers'], learning_rate=network_cfg['learning_rate'])
    
    # Use a dense grid as a consistent validation set
    x_val = np.linspace(plot_cfg['x_min'], plot_cfg['x_max'], 100).reshape(-1, 1)
    y_val = interpolated_function(x_val).reshape(-1, 1)
    
    print(f"▶ Training network for {network_cfg['epochs']} epochs...")
    history = nnet.train(x_train, y_train, x_val, y_val, epochs=network_cfg['epochs'])
    print("✓ Training complete.")

    # --- 5. Generate Predictions and Activation data ---
    subdivision_size = 1000
    x_pred_space = np.linspace(plot_cfg['x_min'], plot_cfg['x_max'], subdivision_size).reshape(-1, 1)
    y_pred = nnet.evaluate(x_pred_space)
    
    activation_change_points = []
    if plot_cfg['show_activation_changes']:
        activation_change_points = nnet.get_activation_change_points(x_pred_space)
        print(f"Found {len(activation_change_points)} activation changes.")


    # --- 6. Plotting Results ---
    fig, (input_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Neural Network Training Results', fontsize=16)

    # Plot 1: Function Approximation
    input_ax.set_title('Function Approximation')
    input_ax.set_xlim(plot_cfg['x_min'], plot_cfg['x_max'])
    input_ax.set_ylim(plot_cfg['y_min'], plot_cfg['y_max'])
    
    # Plot target function
    input_ax.plot(x_pred_space, interpolated_function(x_pred_space), 'b-', label='Target Function')
    input_ax.scatter(breakpoints_x, breakpoints_y, color='blue', zorder=5)
    
    # Plot network prediction
    input_ax.plot(x_pred_space, y_pred, 'g-', label='NN Prediction')
    
    # Plot training samples if enabled
    if plot_cfg['show_sample_points']:
        input_ax.scatter(x_train, y_train, color='orange', alpha=0.6, label='Training Samples')

    # Plot activation changes if enabled
    if plot_cfg['show_activation_changes']:
        for x_val in activation_change_points:
            input_ax.axvline(x=x_val, color='black', linestyle='dotted', linewidth=1)
        # Custom legend entry for dotted line
        from matplotlib.lines import Line2D
        activation_legend_line = Line2D([0], [0], color='black', linestyle='dotted', label=f'Activation Changes ({len(activation_change_points)})')
        handles, labels = input_ax.get_legend_handles_labels()
        handles.append(activation_legend_line)
        input_ax.legend(handles=handles)
    else:
        input_ax.legend()

    input_ax.set_xlabel('Input (x)')
    input_ax.set_ylabel('Output (y)')
    input_ax.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Loss Curves
    loss_ax.set_title('Training & Validation Loss')
    loss_ax.plot(history.history['loss'], label='Training Loss (MSE)')
    loss_ax.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    loss_ax.set_xlabel('Epochs')
    loss_ax.set_ylabel('Mean Squared Error')
    loss_ax.legend()
    loss_ax.grid(True, linestyle='--', alpha=0.6)

    # --- 7. Save and Finalize ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_cfg['output_filename']
    if 'output' in config and config['output'].get('save_experiment', False):
        os.makedirs('results', exist_ok=True)
        output_path = os.path.join('results', output_path)
        plt.savefig(output_path)
        print(f"✓ Plot saved to '{output_path}'.")
    plt.show()
    


#_____________________________________________________________________________________________/ Main execution block
def main():
    """
    Main function to load config and run the experiment.
    """
    config_path = 'config.toml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return
        
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        run_experiment(config)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()