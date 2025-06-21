# ========================================================================================== #
#                                                                                            #
#                    <<< Représentation sous-apprentissage et sur-apprentissage              #
#                                      à l'aide de polynômes >>>                             #
#                                                                                            #
# ========================================================================================== #
#                                                                                            #
#     Outil pour visualiser le comportement d'un modèle sur-paramétré ou sous-paramétré.     #                                                      
#                                                                                            #
# ========================================================================================== #


#_____________________________________________________________________________________________/ Importations :


import numpy as np
import matplotlib.pyplot as plt


#_____________________________________________________________________________________________/ Génération des données :

def generate_noisy_data(n_points=20, noise_level=3.3):
    x = np.linspace(-5, 5, n_points)
    y_true = x ** 2
    noise = np.random.uniform(-noise_level, noise_level, size=x.shape)
    y_noisy = y_true + noise
    return x, y_noisy, y_true

def fit_polynomial(x, y, degree):
    X = np.vander(x, N=degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def evaluate_polynomial(coeffs, x_vals):
    X = np.vander(x_vals, N=len(coeffs), increasing=True)
    return X @ coeffs

#_____________________________________________________________________________________________/ Main :

def main() :
    x, y_noisy, y_true = generate_noisy_data()
    x_dense = np.linspace(-5, 5, 400)
    x_true_eval = np.linspace(-5, 5, 100)
    y_true_eval = x_true_eval ** 2
    degrees_main = [0, 2, 5, 10, 15, 20]

    # Première figure : Affichage du polynôme obtenu pour les degrés choisis dans "degrees_main".

    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, d in enumerate(degrees_main):
        coeffs = fit_polynomial(x, y_noisy, d)
        y_fit = evaluate_polynomial(coeffs, x_dense)
        y_pred_on_x = evaluate_polynomial(coeffs, x)
        y_pred_on_true_sample = evaluate_polynomial(coeffs, x_true_eval)
        loss_emp = np.mean((y_pred_on_x - y_noisy) ** 2)
        loss_true = np.mean((y_pred_on_true_sample - y_true_eval) ** 2)
        ax = axes[i]
        ax.scatter(x, y_noisy, color='red', label='Données')
        ax.plot(x_dense, x_dense ** 2, label='$y = x^2$', linestyle='--', color='gray')
        ax.plot(x_dense, y_fit, color='blue', label='Prédiction')
        ax.set_title(f"$d = {d}$ | Emp. : ${loss_emp:.2f}$ | Pop. : ${loss_true:.2f}$")
        ax.legend()
        ax.set_ylim(-5, 25)
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Seconde figure : Affichage du risque en fonction du degré du polynôme.

    degrees_all = range(0, 21)
    empirical_losses = []
    true_losses = []
    for d in degrees_all:
        coeffs = fit_polynomial(x, y_noisy, d)
        y_pred_on_x = evaluate_polynomial(coeffs, x)
        y_pred_on_true_sample = evaluate_polynomial(coeffs, x_true_eval)
        emp_loss = np.mean((y_pred_on_x - y_noisy) ** 2)
        true_loss = np.mean((y_pred_on_true_sample - y_true_eval) ** 2)
        empirical_losses.append(emp_loss)
        true_losses.append(true_loss)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(degrees_all, empirical_losses, marker='o', label='Risque empirique', color='red')
    ax2.plot(degrees_all, true_losses, marker='s', label='Risque en population', color='blue')
    ax2.set_xlabel("Degré du polynôme")
    ax2.set_ylabel("Erreur quadratique moyenne")
    ax2.set_yscale('log')
    ax2.set_title("Comparaison des pertes")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__" :
    main()
