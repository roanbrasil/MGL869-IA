import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Définir les chemins d'accès pour les ensembles d'entraînement et de test
train_path =  ''#mettre ici chemin vers dossier train
validation_path =  ''#mettre ici chemin vers dossier validation

# Fonction pour charger les chemins des images et leurs labels
def load_dataset(data_dir):
    categories = os.listdir(data_dir)
    data = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            data.append((file_path, category))
    return pd.DataFrame(data, columns=['file_path', 'label'])

# Charger les ensembles d'entraînement et de test
train_data = load_dataset(train_path)
test_data = load_dataset(test_path)

# Afficher les premières lignes pour vérifier
print("Aperçu des données d'entraînement :")
print(train_data.head())
print("\nAperçu des données de test :")
print(test_data.head())

# Statistiques descriptives pour le DataFrame entier
print("\nStatistiques descriptives pour les données d'entraînement :")
print(train_data.describe(include='all'))

print("\nStatistiques descriptives pour les données de test :")
print(test_data.describe(include='all'))

# Distribution des classes dans l'ensemble d'entraînement
print("\nDistribution des classes dans l'ensemble d'entraînement :")
print(train_data['label'].value_counts())

# Distribution des classes dans l'ensemble de test
print("\nDistribution des classes dans l'ensemble de test :")
print(test_data['label'].value_counts())

# Visualiser la distribution des classes dans l'ensemble d'entraînement
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='label')
plt.title('Distribution des classes dans l\'ensemble d\'entraînement')
plt.xticks(rotation=45)
plt.show()

# Visualiser la distribution des classes dans l'ensemble de test
plt.figure(figsize=(10, 6))
sns.countplot(data=test_data, x='label')
plt.title('Distribution des classes dans l\'ensemble de test')
plt.xticks(rotation=45)
plt.show()
