import os
import argparse
from pathlib import Path

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--files_root", action="store", dest="files_root", type=str)
args = arg_parser.parse_args()

file_root = Path(args.files_root)

train_path = file_root / "seg_train"  # mettre ici chemin vers dossier train
validation_path = file_root / "seg_test"  # mettre ici chemin vers dossier validation
test_path = file_root / "seg_pred"  # mettre ici chemin vers dossier test


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
