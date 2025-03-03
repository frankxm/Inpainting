# Inpainting

Pour supprimer les pixels nous utilisons L'inpainting (interpolation intelligente des pixels d'image). En utilisant la méthode de suppression d'objets par retouche basée sur des exemples. Ce code utilise des techniques de traitement d'images pour détecter des motifs spécifiques dans une image en utilisant un masque prédéfini, et il utilise la compilation à la volée avec Numba pour accélérer certaines parties critiques du code.

Ce projet réalisé en python3 est l'implémentation de [cet algorithme](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf)

J'utilise la librairie OpenCV et Numpy.

# Traitement d'Image - Inpainting

## Identification du front de remplissage (FillFront)

La fonction `IdentifyTheFillFront` identifie les bords du masque (zones à traiter) en utilisant des filtres de convolution (Laplacien et gradients X/Y). Elle calcule les normales aux bords et retourne les positions des pixels à traiter (`dOmega`) et leurs vecteurs normaux (`normale`).

## Calcul des priorités pour chaque pixel

La fonction `calculPriority` dans le module `priorities` calcule la priorité pour chaque pixel de bordure en combinant la confiance du pixel et les données calculées à partir des gradients. Cette priorité permet de déterminer quel patch doit être rempli en premier.

## Sélection du patch optimal

Le module `bestpatch` est utilisé pour sélectionner le patch optimal qui sera utilisé pour remplir la zone manquante. La fonction `calculPatch` recherche le patch avec la meilleure correspondance structurée en calculant la somme des carrés des différences (SSD) entre les patches voisins non masqués et le patch cible.

## Mise à jour de l'image

La fonction `update` applique le patch sélectionné sur l'image en remplaçant les pixels manquants par ceux du patch source. Elle met également à jour les cartes de confiance, de source et de masque pour refléter les modifications apportées à l'image. Les gradients, ainsi que d'autres informations nécessaires, sont également ajustés.

## Traitement itératif

Ce processus est répété de manière itérative pour chaque pixel de bordure identifié jusqu'à ce que toute la zone manquante de l'image soit remplie.

## Processus de traitement

![Processus de traitement](images_demo/processus_inpainting.png)
