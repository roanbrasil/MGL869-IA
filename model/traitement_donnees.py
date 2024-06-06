import os
import cv2
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Augmentation des données
def create_directory_structure(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for category in ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']:
            category_dir = os.path.join(split_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

def augment_and_save_images(image_path, save_dir, num_augmented=2, img_width=150, img_height=150):
    datagen = ImageDataGenerator(
        rotation_range=30,             # Rotation
        width_shift_range=0.2,         # Décalage horizontal
        height_shift_range=0.2,        # Décalage vertical
        shear_range=0.2,               # Cisaillement
        zoom_range=0.2,                # Zoom
        brightness_range=[0.8, 1.2],   # Augmenter la luminosité
        horizontal_flip=True,          # Flip horizontal
        fill_mode='nearest',           # Mode de remplissage pour les nouvelles zones
        rescale=1.0/255.0              #Normaliser
    )

    # Lire et redimensionner l'image d'origine
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, 0)  # Ajouter une dimension pour le générateur

    # Copier l'image originale
    base_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(save_dir, base_name))

    # Générer et enregistrer les images augmentées
    for i, batch in enumerate(datagen.flow(img, batch_size=1, save_to_dir=save_dir, save_prefix=os.path.splitext(base_name)[0] + '_aug', save_format='jpg')):
        if i >= num_augmented:
            break

def augment_dataset(dataset_path, save_base_dir, split, img_width=150, img_height=150, num_augmented=2):
    categories = os.listdir(dataset_path)
    for category in tqdm(categories):
        category_path = os.path.join(dataset_path, category)
        save_dir = os.path.join(save_base_dir, split, category)
        for file in os.listdir(category_path):
            if file.endswith('.jpg'):
                file_path = os.path.join(category_path, file)
                augment_and_save_images(file_path, save_dir, num_augmented, img_width, img_height)

# Chemins vers les ensembles d'entraînement, de validation et de test
train_path = "_train"
validation_path = ""  # Assumed to be the same as train for demonstration
test_path = ""

save_base_dir = "/content/augmented_dataset"

create_directory_structure(save_base_dir)

# Augmenter les données dans les ensembles d'entraînement, de validation et de test
augment_dataset(train_path, save_base_dir, 'train')
augment_dataset(validation_path, save_base_dir, 'validation')
augment_dataset(test_path, save_base_dir, 'test')
