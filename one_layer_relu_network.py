# ========================================================================================== #
#                                                                                            #
#                    <<< Réseau de neurones à une couche cachée >>>                          #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#       Outil pour visualiser le comportement d'un réseau de neurone avec une unique         #
#    couche cachée et gestion manuelle des poids et biais                                    #                                                      
#                                                                                            #
#    Options :                                                                               #
#      - Génére un ensemble de points                                                        #
#      - Entrainement du réseau manuellement                                                 #
#      - Affichage des changement de zones d'activation 2D                                   #
#      - Affichage du paysage 3D de la fonction de prédiction f_{\theta}                     #
#                                                                                            #
# ========================================================================================== #



#_____________________________________________________________________________________________/ Importation :

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


#_____________________________________________________________________________________________/ Réseau de Neurone :

class OneLayerReLU:
    def __init__(self, n):
        ''' Initialise un réseau ReLU d'architecture (2, n, 1)'''
        self.n = n
        self.W1 = np.random.rand(n, 2)
        self.b1 = np.random.rand(n)
        self.W2 = np.random.rand(1, n)
        self.b2 = np.random.rand(1)

    def ReLU(self, x):
        return np.maximum(x, 0)

    def forward_propagation(self, input):
        layer1 = self.ReLU(self.W1.dot(input) + self.b1)
        layer2 = self.W2.dot(layer1) + self.b2
        return layer2
    

#_____________________________________________________________________________________________/ Données d'entraînement :

def generate_points(n):
    ''' Crée un dataset de taille n qui avec répartition uniforme dans [-1, 1] x [-1, 1] (et les étiquette selon leur position)'''
    points = []
    for _ in range(n):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        coords = np.array([x, y])
        # Correspond à la zone de l'espace délimitée par les segments noirs
        y_lower = 0.5 * x - 0.2
        y_upper = -0.5 * x + 0.2
        label = 1 if y_lower < y < y_upper else 0
        points.append((coords, label))
    return points


#_____________________________________________________________________________________________/ Affichage 2D :

def setup_2d_plot(nnet, points):
    '''
    Initialise une figure matplotlib 2D avec les éléments nécessaires : nuages de points, droites de séparation,
    et lignes représentant les neurones du réseau. Configure l’échelle, la légende, et les styles de tracé.
    '''
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.5)
    scatter_blue = ax.scatter([], [], color='blue', alpha=0.5, label=r'Sorties du réseau $\leq 0.5$')
    scatter_red = ax.scatter([], [], color='red', alpha=0.5, label='Sorties du réseau $> 0.5$')
    lines = []
    x_vals_line = np.linspace(-1, 0.4, 200)
    ax.plot(x_vals_line, 0.5 * x_vals_line - 0.2, 'k-', label='y = 0.5x - 0.2')
    ax.plot(x_vals_line, -0.5 * x_vals_line + 0.2, 'k-', label='y = -0.5x + 0.2')
    x_vals_net = np.linspace(-1, 1, 200)
    for i in range(nnet.n):
        w = nnet.W1[i]
        b = nnet.b1[i]
        if abs(w[1]) > 1e-6:
            y_vals = -(w[0] * x_vals_net + b) / w[1]
            (line,) = ax.plot(x_vals_net, y_vals, color='green', linestyle='-', alpha=0.7)
        else:
            x_val = -b / w[0]
            (line,) = ax.plot([x_val]*2, [-1, 1], color='green', linestyle='-', alpha=0.7)
        lines.append(line)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Réseau de neurones - ajustable")
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig, ax, scatter_red, scatter_blue, lines

def update_network_params(sliders, nnet):
    '''
    Met à jour les paramètres du réseau (poids et biais) à partir des valeurs des sliders, avant d'appeler une mise à jour graphique.
    '''
    for slider, info in sliders:
        if info[0] == 'W1':
            i, j = info[1], info[2]
            nnet.W1[i, j] = slider.val
        elif info[0] == 'b1':
            i = info[1]
            nnet.b1[i] = slider.val

def create_sliders(n, nnet):
    '''
    Crée des sliders interactifs pour modifier dynamiquement les poids W1[i,j] et les biais b1[i]
    du réseau de neurones. Retourne la liste des sliders avec leur métadonnée associée.
    '''
    sliders = []
    start_y = 0.95
    step_y = 0.03
    slider_height = 0.02
    slider_width = 0.25

    for i in range(n):
        for j in range(2):
            ax_slider = plt.axes([0.05, start_y - step_y * (i * 2 + j), slider_width, slider_height])
            slider = Slider(ax_slider, f'W1[{i},{j}]', -5.0, 5.0, valinit=nnet.W1[i, j])
            sliders.append((slider, ('W1', i, j)))

        ax_slider = plt.axes([0.40, start_y - step_y * i - 0.6, slider_width, slider_height])
        slider = Slider(ax_slider, f'b1[{i}]', -5.0, 5.0, valinit=nnet.b1[i])
        sliders.append((slider, ('b1', i)))

    return sliders

def update_plot(nnet, points, scatter_red, scatter_blue, lines):
    '''
    Met à jour l'affichage des points et des hyperplans du réseau après un changement des poids/biais.
    '''
    red_points = []
    blue_points = []
    for p, _ in points:
        output = nnet.forward_propagation(p)
        (red_points if output[0] > 0.5 else blue_points).append(p)
    red_points = np.array(red_points)
    blue_points = np.array(blue_points)
    scatter_red.set_offsets(red_points if len(red_points) > 0 else np.empty((0, 2)))
    scatter_blue.set_offsets(blue_points if len(blue_points) > 0 else np.empty((0, 2)))
    x_vals = np.linspace(-1, 1, 200)
    for i in range(nnet.n):
        w = nnet.W1[i]
        b = nnet.b1[i]
        if abs(w[1]) > 1e-6:
            y_vals = -(w[0] * x_vals + b) / w[1]
            lines[i].set_data(x_vals, y_vals)
        else:
            x_val = -b / w[0]
            lines[i].set_data([x_val]*2, [-1, 1])
    plt.gcf().canvas.draw_idle()


#_____________________________________________________________________________________________/ Affichage 3D :

def plot_3d_surface(nnet):
    ''' 
    Affiche la surface 3D de sortie du réseau de neurones. Utilisé pour visualiser le comportement du réseau.
    '''
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [nnet.forward_propagation(np.array([xi, yi]))[0] for xi, yi in zip(row_x, row_y)]
        for row_x, row_y in zip(X, Y)
    ])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    x_vals = np.linspace(-3, 3, 200)
    for i in range(nnet.n):
        w = nnet.W1[i]
        b = nnet.b1[i]
        if abs(w[1]) > 1e-6:
            y_vals = -(w[0] * x_vals + b) / w[1]
        else:
            x_vals = np.full(200, -b / w[0])
            y_vals = np.linspace(-3, 3, 200)

        z_vals = np.array([
            nnet.forward_propagation(np.array([x, y]))[0]
            for x, y in zip(x_vals, y_vals)
        ])
        ax.plot(x_vals, y_vals, z_vals, color='green', linewidth=1.0, alpha=0.7)
    plt.show()

#_____________________________________________________________________________________________/ Main :

def main():
    n = 3
    nnet = OneLayerReLU(n)
    points = generate_points(500)

    fig, ax, scatter_red, scatter_blue, lines = setup_2d_plot(nnet, points)
    sliders = create_sliders(n, nnet)

    def update(val=None):
        update_network_params(sliders, nnet)
        update_plot(nnet, points, scatter_red, scatter_blue, lines)

    for slider, _ in sliders:
        slider.on_changed(update)

    update()
    plt.show()

    plot_3d_surface(nnet)

if __name__ == "__main__" :
    main()
