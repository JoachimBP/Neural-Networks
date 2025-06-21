# ========================================================================================== #
#                                                                                            #
#                                 <<< Réseau de neurones >>>                                 #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#    Outil pour créer et entraîner un réseau de neurone ReLU                                 #
#                                                                                            #
#    Options :                                                                               #
#      - Créer la fonction objectif manuellement                                             #
#      - Générer des données avec un bruit aléatoire                                         #
#      - Entrainer le réseau et afficher la fonction obtenue                                 #
#      - Affichage des changement de pattern d'activation                                    #
#      - Afficher/masquer les données d'entraînement                                         #
#                                                                                            #
# ========================================================================================== #





#_____________________________________________________________________________________________/ Importation :


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.interpolate import interp1d
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

#_____________________________________________________________________________________________/ Constantes :

LIMITS = (0, 20, 0, 20)
SUBDIVISION_SIZE = 1000


#_____________________________________________________________________________________________/ Variables globales :

activation_lines = []
activation_lines_visible = False
sample_scatter_visible = True


#_____________________________________________________________________________________________/ Préparation de la grille :

fig, (input_ax, loss_ax) = plt.subplots(1, 2, figsize=(10, 5))
scatter = input_ax.scatter([], [], color='blue', label='_nolegend_')
sample_scatter = input_ax.scatter([], [], color='orange', label='Échantillon')
input_ax.set_xlim(LIMITS[0], LIMITS[1])
input_ax.set_ylim(LIMITS[2], LIMITS[3])
fig.subplots_adjust(right=0.8)
activation_legend_line = Line2D([], [], color='black', linestyle='dotted', label="Changement d'activation")
plt.rcParams['font.size'] = 14


#_____________________________________________________________________________________________/ Boutons :

button_start_ax = plt.axes([0.82, 0.8, 0.15, 0.045])
button_start = Button(button_start_ax, 'Start') 
button_undo_ax = plt.axes([0.82, 0.75, 0.15, 0.045])
button_undo = Button(button_undo_ax, 'Retour')
toggle_activation_ax = plt.axes([0.82, 0.70, 0.15, 0.045])
toggle_activation_button = Button(toggle_activation_ax, 'Activation visible')
toggle_sample_ax = plt.axes([0.82, 0.65, 0.15, 0.045])
toggle_sample_button = Button(toggle_sample_ax, 'Échantillon visible')


#_____________________________________________________________________________________________/ Sliders :

sample_slider_ax = plt.axes([0.93, 0.1, 0.03, 0.4])
sample_slider = Slider(sample_slider_ax, 'Taille\néchantillon', 1, 100, valinit=10, orientation='vertical', valstep=1)
noise_slider_ax = plt.axes([0.88, 0.1, 0.03, 0.4])
noise_slider = Slider(noise_slider_ax, 'Bruit', 0.0, 1.5, valinit=0.0, orientation='vertical', valstep=0.1)
epoch_slider_ax = plt.axes([0.83, 0.1, 0.03, 0.4])
epoch_slider = Slider(epoch_slider_ax, 'Epochs', 10, 1000, valinit=200, orientation='vertical', valstep=10)


#_____________________________________________________________________________________________/ Affichage du nombre de zones affines :

affine_zone_ax = plt.axes([0.82, 0.60, 0.15, 0.045])
affine_zone_text = affine_zone_ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=10)
affine_zone_ax.set_axis_off()


#_____________________________________________________________________________________________/ Fonction interpolée & prédiciton  :

interpolated_function = None
interpolated_curve, = input_ax.plot([], [], 'b-', label='Fonction cible')
nn_curve, = input_ax.plot([], [], 'g-', label='Prédiction')


#_____________________________________________________________________________________________/ Affichage :

def update_line():
    """
    Met à jour la courbe interpolée reliant les points affichés dans le nuage principal (`scatter`).

    - Trie les points selon l'abscisse,
    - Reconstruit une fonction d'interpolation linéaire `interpolated_function`,
    - Met à jour la courbe bleue affichée (`interpolated_curve`),
    - Désactive la courbe si le nombre de points est insuffisant (< 2).
    """

    global interpolated_function
    points = scatter.get_offsets()
    if len(points) >= 2:
        sorted_points = points[np.argsort(points[:, 0])]
        x_vals = sorted_points[:, 0]
        y_vals = sorted_points[:, 1]

        interpolated_function = interp1d(x_vals, y_vals, kind='linear', fill_value="extrapolate")
        x_curve = np.linspace(LIMITS[0], LIMITS[1], SUBDIVISION_SIZE)
        y_curve = interpolated_function(x_curve)
        interpolated_curve.set_data(x_curve, y_curve)

    else:
        interpolated_curve.set_data([], [])
        interpolated_function = None

    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes == input_ax:
        new_point = np.array([[event.xdata, event.ydata]])
        offsets = scatter.get_offsets()
        new_offsets = np.vstack([offsets, new_point])
        scatter.set_offsets(new_offsets)
        fig.canvas.draw_idle()
        update_line()

def undo_last_point(event):
    """
    Supprime le dernier point ajouté dans le nuage principal (`scatter`),
    puis met à jour la courbe d'interpolation via `update_line()`.
    """
    offsets = scatter.get_offsets()
    if len(offsets) > 0:
        new_offsets = offsets[:-1]
        scatter.set_offsets(new_offsets)
        update_line()

def generate_random_points(val=None):
    """
    Génère des points d'apprentissage aléatoires à partir de la courbe interpolée :
        - Les abscisses sont choisies uniformément,
        - Les ordonnées sont interpolées et bruitées (gaussien),
    """
    n = sample_slider.val
    noise_level = noise_slider.val
    x_vals = np.random.uniform(LIMITS[0], LIMITS[1], size=n)
    y_vals = interpolated_function(x_vals)
    y_vals += np.random.normal(0, noise_level, size=n)
    new_points = np.column_stack((x_vals, y_vals))
    sample_scatter.set_offsets(new_points)
    fig.canvas.draw_idle()

def toggle_sample_scatter(event=None):
    """
    Affiche ou masque dynamiquement le nuage de points d'apprentissage aléatoires (`sample_scatter`).
    Modifie aussi le texte du bouton associé.
    """
    global sample_scatter_visible
    sample_scatter_visible = not sample_scatter_visible
    sample_scatter.set_visible(sample_scatter_visible)
    toggle_sample_button.label.set_text('Éhantillon caché' if sample_scatter_visible else 'Échantillon visible')
    fig.canvas.draw_idle()


#_____________________________________________________________________________________________/ Neural network :

class ReLU_network :
    def __init__(self, layers) :
        ''' Initialise un réseau de neurone d'architecture (layers)'''
        self.input_size = layers[0]
        self.output_size = layers[-1]
        self.size = len(layers)

        self.nnet = Sequential()
        self.nnet.add(Input(shape=(self.input_size,)))
        for size in layers[2:-1]:
            self.nnet.add(Dense(size, activation='relu'))
        self.nnet.add(Dense(self.output_size, activation='linear'))
        optimizer = Adam(learning_rate=0.05)
        self.nnet.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train(self, X_train, Y_train, X_test, Y_test, epochs) :
        return self.nnet.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

    def evaluate(self, X) :
        return self.nnet.predict(X)

    def get_activation_change_points(self, x):
        ''' Permet d'obtenir les changement de zones d'activation du réseau'''
        current_output = tf.convert_to_tensor(x, dtype=tf.float32)
        binary_activations = []
        for layer in self.nnet.layers:
            current_output = layer(current_output)
            if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
                binary_activations.append(tf.cast(current_output > 0, tf.int8))
        signatures = tf.concat([tf.reshape(act, (act.shape[0], -1)) for act in binary_activations], axis=1).numpy()
        change_indices = np.any(np.diff(signatures, axis=0), axis=1)
        change_points = x[1:][change_indices].flatten()
        return change_points

    def __str__(self) :
        self.nnet.summary()
        return ""


#_____________________________________________________________________________________________/ Neural network display management :

def train_and_plot_nn(event=None):
    """
    Entraîne le réseau de neurones sur l'échantillons visibles, puis met à jour dynamiquement :
    - la courbe de prédiction du réseau,
    - le graphique du risque (empirique et en population).

    Utilise le nombre d'épochs indiqué par le slider associé.
    Ignore l'appel si aucun échantillon n'est affiché.
    """
    offsets = sample_scatter.get_offsets()
    if len(offsets) == 0:
        print("No data to train.")
        return
    x_train = offsets[:, 0].reshape(-1, 1)
    y_train = offsets[:, 1].reshape(-1, 1)
    x_test = np.linspace(LIMITS[0], LIMITS[1], 100)
    y_test = interpolated_function(x_test)
    epochs = int(epoch_slider.val)
    history = nnet.train(x_train, y_train, x_test, y_test, epochs)
    loss_ax.clear()
    loss_ax.plot(history.history['loss'], label='Risque empirique', color='blue')
    loss_ax.plot(history.history['val_loss'], label='Risque en pop', color='orange')
    loss_ax.set_xlabel('Epochs')
    loss_ax.set_ylabel('Risque')
    loss_ax.legend(loc='upper right')
    x_pred = np.linspace(LIMITS[0], LIMITS[1], SUBDIVISION_SIZE).reshape(-1, 1)
    y_pred = nnet.evaluate(x_pred)
    nn_curve.set_data(x_pred.flatten(), y_pred.flatten())
    fig.canvas.draw_idle()

def toggle_activation_lines(event=None):
    """
    Affiche ou masque les lignes verticales représentant les points de changement de zones affines (i.e. de pattern d'activation)
    Met également à jour le texte affichant le nombre de zones affines.
    """
    global activation_lines, activation_lines_visible
    if activation_lines_visible:
        for line in activation_lines:
            line.remove()
        activation_lines.clear()
        activation_lines_visible = False
        toggle_activation_button.label.set_text('Activation visible')
    else:
        x_vals = np.linspace(LIMITS[0], LIMITS[1], SUBDIVISION_SIZE).reshape(-1, 1)
        change_points = nnet.get_activation_change_points(x_vals)
        for x in change_points:
            line = input_ax.axvline(x=x, color='black', linestyle='dotted', linewidth=1)
            activation_lines.append(line)
        activation_lines_visible = True
        toggle_activation_button.label.set_text('Activation cachée')
        num_affine_zones = len(change_points) + 1
        affine_zone_text.set_text(f'Affine zones: {num_affine_zones}')
    fig.canvas.draw_idle()

def display_clean_figure():
    """
    Affiche une figure matplotlib avec :
    - la fonction cible (courbe bleue),
    - la prédiction du réseau (courbe verte),
    - les échantillons utilisés pour l'entraînement,
    - les lignes de changement d'activation,
    - le graphique du risque.

    En particulier, les sliders et boutons n'apparisent plus
    """
    fig2, (input_ax2, loss_ax2) = plt.subplots(1, 2, figsize=(10, 5))
    input_ax2.set_xlim(LIMITS[0], LIMITS[1])
    input_ax2.set_ylim(LIMITS[2], LIMITS[3])
    input_ax2.plot(*interpolated_curve.get_data(), 'b-', label='Fonction cible')
    input_ax2.plot(*nn_curve.get_data(), 'g-', label='Prédiction')
    if sample_scatter_visible:
        input_ax2.scatter(*sample_scatter.get_offsets().T, color='orange', label='Échantillon')
    input_ax2.scatter(*scatter.get_offsets().T, color='blue')
    if activation_lines_visible:
        for line in activation_lines:
            input_ax2.axvline(x=line.get_xdata()[0], color='black', linestyle='dotted')
    input_ax2.legend(loc='upper right')
    for line in loss_ax.get_lines():
        loss_ax2.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())

    loss_ax2.set_xlabel('Epochs')
    loss_ax2.set_ylabel('Risque')
    loss_ax2.legend(loc='upper right')
    plt.show()

def on_close(event):
    display_clean_figure()


#_____________________________________________________________________________________________/ Main :

def main():
    global nnet 
    nnet = ReLU_network([1, 50, 50, 1])
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)
    button_undo.on_clicked(undo_last_point)
    button_start.on_clicked(train_and_plot_nn)
    button_undo.on_clicked(generate_random_points)
    noise_slider.on_changed(generate_random_points)
    sample_slider.on_changed(generate_random_points)
    toggle_activation_button.on_clicked(toggle_activation_lines)
    toggle_sample_button.on_clicked(toggle_sample_scatter)
    input_ax.legend(handles=[sample_scatter, interpolated_curve, nn_curve,activation_legend_line], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
