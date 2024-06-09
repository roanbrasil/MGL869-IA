import os
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--files_root", action="store", dest="files_root", type=str)
args = arg_parser.parse_args()

file_root = Path(args.files_root)

# Définir les chemins d'accès pour les ensembles d'entraînement et de test
train_path = file_root / "seg_train"  # mettre ici chemin vers dossier train
validation_path = file_root / "seg_test"  # mettre ici chemin vers dossier validation


# Fonction pour charger les chemins des images et leurs labels
def load_dataset(data_dir):
    categories = os.listdir(data_dir)
    data = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            data.append((file_path, category))
    return pd.DataFrame(data, columns=["file_path", "label"])


# Charger les ensembles d'entraînement et de test
train_data = load_dataset(train_path)
test_data = load_dataset(validation_path)

# Afficher les premières lignes pour vérifier
print("Aperçu des données d'entraînement :")
print(train_data.head())
print("\nAperçu des données de test :")
print(test_data.head())

# Statistiques descriptives pour le DataFrame entier
print("\nStatistiques descriptives pour les données d'entraînement :")
print(train_data.describe(include="all"))

print("\nStatistiques descriptives pour les données de test :")
print(test_data.describe(include="all"))

# Distribution des classes dans l'ensemble d'entraînement
print("\nDistribution des classes dans l'ensemble d'entraînement :")
print(train_data["label"].value_counts())

# Distribution des classes dans l'ensemble de test
print("\nDistribution des classes dans l'ensemble de test :")
print(test_data["label"].value_counts())

# Visualiser la distribution des classes dans l'ensemble d'entraînement
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x="label")
plt.title("Distribution des classes dans l'ensemble d'entraînement")
plt.xticks(rotation=45)
plt.show()

# Visualiser la distribution des classes dans l'ensemble de test
plt.figure(figsize=(10, 6))
sns.countplot(data=test_data, x="label")
plt.title("Distribution des classes dans l'ensemble de test")
plt.xticks(rotation=45)
plt.show()
