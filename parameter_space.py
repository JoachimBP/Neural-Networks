# ========================================================================================== #
#                                                                                            #
#                         <<< Réseau de neurones (1, 1, 1) >>>                               #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#       Outil pour visualiser le comportement d'un réseau de neurones de paramétrisation     #
#   (1, 1, 1) durant l'apprentissage pour différentes initialisation de $\theta$.            #
#                                                                                            #
#    Remarque :                                                                              #
#      - Les images obtenues sont stockées dans le dossier "/frames"                         #
#                                                                                            #
# ========================================================================================== #



#_____________________________________________________________________________________________/ Importation :

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


#_____________________________________________________________________________________________/ ImportChoix du dossier :

output_dir = "/frames"
os.makedirs(output_dir, exist_ok=True)

#_____________________________________________________________________________________________/ Réseau de neurones :

class ReLU_Network_111:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
        self.learning_rate = 0.1

    def ReLU(self, x):
        return np.maximum(x, 0)

    def ReLU_derivative(self, x):
        return (x > 0).astype(float)

    def forward_propagation(self, x):
        self.x = x
        self.z1 = self.w1 * x + self.b1
        self.a1 = self.ReLU(self.z1)
        self.z2 = self.w2 * self.a1 + self.b2
        self.a2 = self.z2
        return self.a2

    def forward_propagation_inputs(self, inputs):
        return np.array([self.forward_propagation(x) for x in inputs])

    def backward_propagation(self, inputs, targets):
        for x, y in zip(inputs, targets):
            self.forward_propagation(x)
            dL_da2 = 2*(self.a2 - y)
            da1_dz1 = self.ReLU_derivative(self.z1)
            dL_dw2 = dL_da2*self.a1
            dL_db2 = dL_da2
            dL_dw1 = dL_da2*self.w2*da1_dz1*x
            dL_db1 = dL_da2*self.w2*da1_dz1
            self.w1 -= self.learning_rate*dL_dw1
            self.b1 -= self.learning_rate*dL_db1
            self.w2 -= self.learning_rate*dL_dw2
            self.b2 -= self.learning_rate*dL_db2

    def compute_loss(self, inputs, targets):
        predictions = self.forward_propagation_inputs(inputs)
        return np.mean((predictions - targets) ** 2)



def main() :

    #_____________________________________________________________________________________________/ Initialisation aléatoire des réseaux :

    Networks = [ReLU_Network_111(np.random.normal(0, 2), np.random.normal(0, 2), np.random.normal(0, 2), np.random.normal(0, 2)) for _ in range(1000)]


    #_____________________________________________________________________________________________/ Choix des données d'entaînement :

    inputs = np.array([0, 1, 2])
    targets = np.array([1.5, 1.1, 0.4])


    #_____________________________________________________________________________________________/ Entaînement et enregistrement des frames :

    v = np.array([1, 1, 1])
    for step in range(0, 500, 5):
        w1s = np.array([n.w1 for n in Networks])
        b1s = np.array([n.b1 for n in Networks])
        losses = np.array([n.compute_loss(inputs, targets) for n in Networks])
        outs = np.array([n.forward_propagation_inputs(inputs) for n in Networks])
        v = np.array([1, 1, 1])
        v = v / np.linalg.norm(v)
        u1 = np.array([-1, 1, 0])
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.array([1, 1, -2])
        u2 = u2 / np.linalg.norm(u2)
        outs_centered = outs - np.dot(outs, v)[:, np.newaxis] * v
        outs_proj_2d = np.stack([-np.dot(outs_centered, u2), -np.dot(outs_centered, u1)], axis=1)



        fig = plt.figure(figsize=(20, 6))

        #_____________________________________________________________________________________________/ Espace des paramètres (2D) :

        ax1 = fig.add_subplot(1, 3, 1)
        sc1 = ax1.scatter(w1s, b1s, c=losses, cmap='RdYlGn_r', s=10)
        x_vals = np.linspace(-5, 5, 200)
        ax1.plot(x_vals, -x_vals, 'r-', label='$w_1 + b_1 = 0$')
        ax1.plot(x_vals, -2 * x_vals, 'r-', label='$2w_1 + b_1 = 0$')
        ax1.plot(x_vals, 0 * x_vals, 'r-', label='$b_1 = 0$')
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        plt.text(-2, -3, r"$\mathcal{U}^X_1$", fontsize=20, color='red',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))   
        plt.text(-4, 2, r"$\mathcal{U}^X_2$", fontsize=20, color='red',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        plt.text(-3, 3.8, r"$\mathcal{U}^X_3$", fontsize=20, color='red',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        plt.text(2.5, 2, r"$\mathcal{U}^X_4$", fontsize=20, color='red',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        plt.text(2.5, -1, r"$\mathcal{U}^X_5$", fontsize=20, color='red',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        plt.text(2.5, -3.8, r"$\mathcal{U}^X_4$", fontsize=20, color='red',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))     
        ax1.set_xlabel("$w_1$")
        ax1.set_ylabel("$b_1$")
        ax1.set_title(f"Epoch {step} - Paramètres $(w_1, b_1)$")
        ax1.legend()
        ax1.grid(True)
        fig.colorbar(sc1, ax=ax1, label="Risque")

        #_____________________________________________________________________________________________/ Espace des sorties (3D) :

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        sc2 = ax2.scatter(outs[:, 0], outs[:, 1], outs[:, 2], c=losses, cmap='RdYlGn_r', s=10)
        ax2.scatter(targets[0], targets[1], targets[2], color='blue', s=100, label='Cible (-0.5, -0.5, 0.5)')
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        ax2.set_zlabel("$z$")
        ax2.set_title("Image de $(0,1,2)$ par le réseau")


        #_____________________________________________________________________________________________/ Espace des sorties prrojetées dans le plan orthogonal à (1, 1, 1) (2D) :


        ax3 = fig.add_subplot(1, 3, 3)
        sc3 = ax3.scatter(outs_proj_2d[:, 0], outs_proj_2d[:, 1], c=losses, cmap='RdYlGn_r', s=20)
        x_vals = np.linspace(-1.5, 1.5, 200)
        ax3.plot(x_vals, -3**0.5*x_vals, 'r--', label=r'$y = -\sqrt{x}$')
        ax3.plot(x_vals, -1/(3**0.5)* x_vals, 'r--', label=r'y = -1/\sqrt{3}x')
        ax3.plot(x_vals, 0 * x_vals, 'r--', label='$y = 0$')
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xlabel("$X$")
        ax3.set_ylabel("$Y$")
        ax3.set_title("Projection dans le plan orthogonal à $(1,1,1)$")
        ax3.grid(True)
        frame_path = os.path.join(output_dir, f"frame_{step+10:03d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)

        #_____________________________________________________________________________________________/ Entraînement des réseaux :

        for nnet in Networks:
            if(step == 200) :
                nnet.learning_rate = 0.05
            for _ in range(5):
                nnet.backward_propagation(inputs, targets)

if __name__ == "__main__" :
    main()
