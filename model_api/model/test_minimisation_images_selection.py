import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from pathlib import Path
import argparse
import cv2
import shutil

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--files_root", action="store", dest="files_root", type=str)
args = arg_parser.parse_args()

model_module_path = Path(os.path.realpath(__file__)).parent
model_name = "cnn2_model_50x50_batch_96_epochs_15"
model_src = model_module_path / f"repository/{model_name}.keras"
file_root = Path(args.files_root)

# Définir les chemins d'accès pour l'ensemble de test
test_dir = test_path = file_root / "seg_pred"  # mettre ici chemin vers dossier test

img_width, img_height = 50, 50


def read_imgs(path: Path, img_width: int, img_height: int) -> pd.DataFrame:
    data = []
    labels = []
    counter = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, file))
                img = cv2.resize(img, (img_width, img_height))  # Resize
                img = img / 255.0  # Normalize
                data.append(img)
                labels.append(counter)
                counter += 1
    df = pd.DataFrame(list(zip(data, labels)), columns=["image", "class"])
    return df


images_set = read_imgs(test_dir, img_width, img_height)

test_images = images_set["image"]
test_labels = images_set["class"]

# Clustering
# flat_test_images = test_images.reshape(len(test_images), -1)
flat_test_images = np.stack(test_images.values).reshape(
    len(test_images), img_width * img_height * 3
)

# KMeans
num_clusters = 6  # You can adjust this number
num_images_per_cluster = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(flat_test_images)
clusters = kmeans.predict(flat_test_images)

# Selection d'échantillon de chaque cluster en fonction de la taille du cluster
sampled_test_images = []
sampled_test_labels = []

images_per_cluster = {}

for cluster in range(num_clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    indices = cluster_indices[:num_images_per_cluster]
    images_per_cluster[cluster] = list(test_labels.iloc[indices])

min_test_dir = model_module_path / "repository/minimisation_test/"


def write_images(src_dir, dest_dir, images_list):
    counter = 0
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".jpg") and counter in images_list:
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                shutil.copy(os.path.join(root, file), dest_dir / file)
            counter += 1


for images_list in images_per_cluster.values():
    write_images(test_dir, min_test_dir, images_list)
