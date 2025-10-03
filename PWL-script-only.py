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
from keras.optimizers import Adam, SGD
from keras.optimizers.schedules import ExponentialDecay
import toml
import os
import shutil
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
        optimizer_name = network_cfg.get('optimizer', 'adam')

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
        
        # Set up the learning rate (either a fixed value or a scheduler)
        lr_schedule = learning_rate
        if scheduler_cfg and scheduler_cfg.get('enabled', False):
            print("  Learning rate scheduler is ENABLED.")

            # Implement piecewise decay (copied from PWL-single-training.py)
            class PiecewiseDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, initial_learning_rate, decay_steps, decay_rate, constant_step):
                    self.initial_learning_rate = initial_learning_rate
                    self.decay_steps = decay_steps
                    self.decay_rate = decay_rate
                    self.constant_step = constant_step

                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    lr_before = self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps))
                    lr_after = self.initial_learning_rate * (self.decay_rate ** (self.constant_step / self.decay_steps))
                    return tf.cond(
                        step < self.constant_step,
                        lambda: lr_before,
                        lambda: lr_after
                    )

            lr_schedule = PiecewiseDecay(
                initial_learning_rate=learning_rate,
                decay_steps=scheduler_cfg.get('decay_steps', 1000),
                decay_rate=scheduler_cfg.get('decay_rate', 0.9),
                constant_step=scheduler_cfg.get('constant_step', 2000)
            )
        
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr_schedule)
        elif optimizer_name == 'sgd':
            momentum = network_cfg.get('momentum', 0.0) # Default to 0 if not specified
            optimizer = SGD(learning_rate=lr_schedule, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Please use 'adam' or 'sgd'.")
        self.nnet.compile(optimizer=optimizer, loss='mse')

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=32, verbose=1):
        """Trains the neural network."""
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=verbose)

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

def run_experiment(config):
    """
    Loads data, runs multiple training loops, and saves the results.
    """
    # --- 1. Load Parameters from Config ---
    exp_cfg = config['experiment']
    target_func_cfg = config['target_function']
    dataset_cfg = config['dataset']
    network_cfg = config['network']
    scheduler_cfg = config.get('learning_rate_scheduler') # Can be None
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
    train_losses = []
    jacobian_ranks = []
    num_activation_regions = []
    num_seen_activation_regions = []
    num_runs = exp_cfg['num_runs']
    
    # A high-resolution space to check for activation changes
    x_pred_space = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 1000).reshape(-1, 1)

    for i in range(num_runs):
        print(f"\n--- Running training {i + 1}/{num_runs} ---")
        
        # Initialize a new network for each run to get new random weights
        nnet = ReLU_network(network_cfg, scheduler_cfg=scheduler_cfg)
        
        history = nnet.train(x_train, y_train, x_val, y_val, epochs=network_cfg['epochs'], verbose=network_cfg.get('verbose', 1))
                
        # Record final training loss
        train_loss = history.history['loss'][-1]
        train_losses.append(train_loss)
        print(f"  Final Training Loss: {train_loss:.6f}")

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

        # Record rank of the Jacobian with training inputs
        rank = nnet.get_rank(x_train)
        jacobian_ranks.append(rank)
        print(f"  Jacobian Rank: {rank}")

    print("\n✓ All training runs complete.")

    # --- 5. Save Results ---
    if output_cfg.get('save_experiment', False):
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_cfg['results_base_dir'], date_time_str + "_" + output_cfg.get('name', 'experiment'))
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save the configuration used for this run by writing the in-memory 'config' object
        config_save_path = os.path.join(output_path, 'config.toml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            toml.dump(config, f)
        
        # Save the results arrays to a compressed NumPy file
        results_file_path = os.path.join(output_path, 'results.npz')
        np.savez_compressed(
            results_file_path,
            train_losses=np.array(train_losses),
            final_losses=np.array(final_losses),
            seen_activation_regions=np.array(num_seen_activation_regions),
            activation_regions=np.array(num_activation_regions),
            jacobian_ranks=np.array(jacobian_ranks)
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