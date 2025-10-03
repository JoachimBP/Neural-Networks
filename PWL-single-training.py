# ========================================================================================== #
#                                                                                            #
#                         <<< Single Neural Network Training & Plot >>>                      #
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
from keras.optimizers import Adam, SGD
from keras.optimizers.schedules import ExponentialDecay
import toml
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

#_____________________________________________________________________________________________/ Custom Callback

class EarlyStopOnLossThreshold(tf.keras.callbacks.Callback):
    """
    A Keras callback to stop training if the validation loss falls below a certain threshold.
    """
    def __init__(self, threshold):
        super(EarlyStopOnLossThreshold, self).__init__()
        self.threshold = threshold
        self.stopped_early = False # Flag to indicate if training was stopped by this callback

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is not None and current_loss < self.threshold:
            print(f"\nEpoch {epoch+1}: Validation loss ({current_loss:.6f}) is below threshold ({self.threshold}). Stopping and retrying.")
            self.model.stop_training = True
            self.stopped_early = True


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
            print(f"  Using RandomNormal bias initializer (mean={bias_mean}, stddev={bias_stddev}).")
            bias_initializer = tf.keras.initializers.RandomNormal(mean=bias_mean, stddev=bias_stddev)
        else:
            print("  Using default 'zeros' bias initializer.")
            bias_initializer = 'zeros'

        self.nnet = Sequential()
        self.nnet.add(Input(shape=(layers[0],)))
        for size in layers[1:-1]:
            self.nnet.add(Dense(size, activation='relu', kernel_initializer='he_normal'))
        self.nnet.add(Dense(layers[-1], activation='linear', kernel_initializer='he_normal'))
        
        # Set up the learning rate (either a fixed value or a scheduler)
        self.initial_learning_rate = learning_rate
        lr_schedule = learning_rate
        if scheduler_cfg and scheduler_cfg.get('enabled', False):
            print("  Learning rate scheduler is ENABLED.")

            #Implement piecewise decay
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

            lr_schedule = PiecewiseDecay(initial_learning_rate=learning_rate,
                                         decay_steps=scheduler_cfg.get('decay_steps', 1000),
                                         decay_rate=scheduler_cfg.get('decay_rate', 0.9),
                                         constant_step=scheduler_cfg.get('constant_step', 2000))
        
        optimizer_name = network_cfg.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr_schedule)
        elif optimizer_name == 'sgd':
            momentum = network_cfg.get('momentum', 0.0) # Default to 0 if not specified
            optimizer = SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=(momentum > 0))
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Please use 'adam' or 'sgd'.")
        self.nnet.compile(optimizer=optimizer, loss='mse')

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size=32, callbacks=None, verbose=1):
        """Trains the neural network."""
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                           validation_data=(X_val, Y_val), verbose=1, callbacks=callbacks)

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

    # --- 3. Generate Dataset ---
    x_data = np.random.uniform(target_func_cfg['x_min'], target_func_cfg['x_max'], size=dataset_cfg['num_samples'])
    y_data = interpolated_function(x_data) + np.random.normal(0, dataset_cfg['noise_std_dev'], size=dataset_cfg['num_samples'])
    
    x_train = x_data.reshape(-1, 1)
    y_train = y_data.reshape(-1, 1)
    print('X_train=', x_train)
    print('Y_train=', y_train)
    
    x_val = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 100).reshape(-1, 1)
    y_val = interpolated_function(x_val).reshape(-1, 1)
    
    print(f"✓ Dataset generated with {dataset_cfg['num_samples']} samples.")

   # --- 4. Initialize and Train the Neural Network ---
    loss_threshold = output_cfg.get('loss_threshold', 0.0)
    run_count = 0

    while True:
        run_count += 1
        print(f"\n▶ Training Run #{run_count} (target loss >= {loss_threshold})...")
        
        # Initialize a new network and the custom callback for each attempt
        nnet = ReLU_network(network_cfg, scheduler_cfg=scheduler_cfg)
        stop_callback = EarlyStopOnLossThreshold(threshold=loss_threshold)
        
        history = nnet.train(x_train, y_train, x_val, y_val, 
                               epochs=network_cfg['epochs'], 
                               batch_size=network_cfg.get('batch_size', 32),
                               callbacks=[stop_callback])
        
        # If training was stopped early, the loss was too low. Continue to the next run.
        if stop_callback.stopped_early:
            continue

        # If we reach here, training completed all epochs without the loss dropping below the threshold.
        # This is the scenario we want to capture.
        final_loss = history.history['val_loss'][-1]
        print(f"  Final Validation Loss: {final_loss:.6f}")
        print(f"✓ Training complete. Found a model with final loss above the threshold.")
        break


    # --- 5. Generate Predictions for Plotting ---
    x_pred_space = np.linspace(target_func_cfg['x_min'], target_func_cfg['x_max'], 1000).reshape(-1, 1)
    y_pred = nnet.evaluate(x_pred_space)
    
    # --- 6. Plotting Results ---
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

    # --- 7. Save and Finalize ---
    if output_cfg.get('save_experiment', False):
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_cfg['results_base_dir'], f"single_run_{date_time_str}")
        
        os.makedirs(output_path, exist_ok=True)
        
        shutil.copy('config.toml', os.path.join(output_path, 'config.toml'))
        
        plot_filename = os.path.join(output_path, 'prediction_vs_target.png')
        plt.savefig(plot_filename)
        print(f"\n✓ Plot saved to '{plot_filename}'.")
        
        # Save model weights and biases
        weights_filename = os.path.join(output_path, 'model.weights.h5')
        nnet.nnet.save_weights(weights_filename)
        print(f"✓ Model weights saved to '{weights_filename}'.")
    
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