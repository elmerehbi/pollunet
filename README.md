# Présentation

Ce dépôt contient une adaptation du réseau de neurones U-net (https://arxiv.org/pdf/1505.04597.pdf), pour une expérimentation sur la détection de pollutions maritimes sur des images satellites (SAR).
Le jeu de données a été fourni par l'entreprise CLS (www.cls.fr/).
Le but n'est pas de produire un front-end utilisable directement, ce travail s'est restreint au côté expérimental. En conséquence, le code tel quel nécessite une configuration précise (emplacement des fichiers, forme des données...) et n'est pas optimisé (parfois très lent)

# Requirement

L'expérimentation s'est faite sur une machine dotée d'un disque dur de 2T, 64GiB de RAM, un processeur Intel Xeon à 32 coeurs et une carte graphique Nvidia GeForce 1080Ti (11GiB de mémoire).
Sur des gros jeux de données, la RAM est utilisée jusqu'à 55GiB.

Les packages Python nécessaires sont h5py, keras, matplotlib, numpy, tensorflow-gpu et leur dépendances (calculées par conda).

# Fonctionnement du code

## Modèles

Chaque modèle de réseau de neurones est dans un fichier différent, qui contient la structure du modèle, le code pour prétraiter les données dont il a besoin, entraîner le réseau et l'appliquer sur les jeux de données de test.
Pour des raisons de simplicité, seul le fichier gmf3.py est commenté (c'est le plus récent, et le plus propre), mais tous les autre fichiers ont une structure similaire.

## Jeu de données

Le jeu de données de base consiste en 204 fichiers netcdf contenant une image SAR de dimensions variables prise autour du détroit de Gibraltar, la segmentation correspondante, les données de vent déduites de l'image et prédite, l'angle d'incidence selon une dimension, et des données de fréquence de pollution et  baythymétrie, qui n'ont pas été utilisées ici.

Il a d'abord été séparé en deux dossiers, train et test, test contenant les 34 images les plus récentes (octobre 2015 à mars 2017) et train les autres (jusqu'à mai 2017). Ces deux dossiers séparent les images d'entraînement des images de test. Des patches en ont ensuite été extraits (img_extract.py).

Sur pc-sc-086, les données se présentent sous la forme suivante:
```
/users/local/h17valen/Deep_learning_pollution/:
|__Database_netcdf: netcdf_dir dans le code
|  |__train: 170 fichiers pour l'entraînement
|  |__test: 34 fichier pour les test
|__data.hdf5: fichier hdf contenant tout (patch extraits, masques, poids, résultats), correspondant à la variable hdf dans le code
|__weights: dossiers contenant les poids des réseaux entraînés, par nom de fichier correspondant (weight_dir dans le code)
   |__4classes: poids des réseaux entraînés pour reconnaître 4 classes, ensuite modifiés pour 3 classes (commit 'retiré la terre')
```

## Résultats

Les résultats sous forme brute sont dans le fichier resultats.

# Entraînement

Les réseaux n'ont pas tous été entraînées sur les même données, les patchs ayant été extraits plusieurs fois, ni de la même façon, certains sur tout, d'autre uniquement sur les pollutions, les derniers sur toutes les pollution plus une partie du reste, qu'on change après plusieurs époques. Cette dernière méthode permettait d'appendre bien les pollutions sous repésentées sans être obligé de leur mettre un poids trop fort, et apprenant ce qui n'est pas de la pollution.
