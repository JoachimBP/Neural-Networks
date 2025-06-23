# Réseaux de neurones

## Contexte :
  Ce projet a été implémenté dans le cadre de mon stage de première année du diplôme du Magistère de Mathématiques de l'ENS Rennes sous la supervision de François Malgouyres de l'Institut de Mathématiques de Toulouse. Ce stage portait sur les réseaux de neurones, les phénomènes de sur-apprentissage, sous-apprentissage, régularisation implicite, géométrie des réseaux de neurones... L'objectif de ces fichiers est de visualiser certains de ces phénomènes via des exemples très simples.
  
## 📁 Contenu du projet

- `one_layer_relu_network.py` : Ce fichier permet de visualiser en 2D et 3D le comportement d’un réseau de neurones simple à une couche cachée avec fonction d’activation ReLU. L’utilisateur peut interagir dynamiquement avec les **poids** et **biais** via des **sliders** afin d’observer l’effet sur la frontière de décision et la surface de sortie.

- `polynomial_models.py` : Ce fichier permet de visualiser le comportement d'un modèle d'apprentissage en fonction du nombre de paramètres pour comprendre le phénomène de sur-apprentissage et sous-apprentissage. L'exemple utilisé est un ensemble de polynômes de diffèrents degrés approximant un nuage de points.

- `piecewise_linear_approximation_via_relu_network.py` : Ce fichier permet de choisir une fonction continue affine par morceau, puis d'entrainer un réseau pour approximer cette fonction. Il est ensuite possible de calculer les changements de pattern d'activation.

- `network_trained_with_polynomial_target.py` : (Inutilisé durant le stage) Ce fichier permet de visualiser la prédiction d'un réseau ReLU entraîné sur une fonction cible polynômiale en fonction du nombre d'epochs.

- `parameter_space` : Ce fichier permet de visualiser l'espace des paramètres pour un réseau d'architecture $(1, 1, 1)$. Il permet en particulier de visualiser l'influence de l'initialisation sur l'apprentissage.

![GIF](parameter_space.gif)
