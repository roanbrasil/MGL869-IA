import os
import random
import shutil


train_path =  ''#mettre ici chemin vers dossier train
validation_path =  ''#mettre ici chemin vers dossier validation
test_path = ''#mettre ici chemin vers dossier test



print("Séparation terminée et répertoires créés avec succès !")
# Fonction pour compter le nombre d'images dans un répertoire
def count_images(directory):
    return sum(len(files) for _, _, files in os.walk(directory))

# Compter le nombre d'images dans les répertoires locaux
num_train_images = count_images(train_path)
num_validation_images = count_images(validation_path)
num_test_images = count_images(test_path)


# Afficher le nombre d'images dans chaque ensemble
print("Nombre d'images dans l'ensemble d'entraînement :", num_train_images)
print("Nombre d'images dans l'ensemble de validation :", num_validation_images)
print("Nombre d'images dans l'ensemble de validation :", num_test_images)
