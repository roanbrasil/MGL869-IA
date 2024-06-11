import os
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load data and preprocess.
# ------------------------------
# Preprocessing 1 - Resize - The CNN requires the images to be in the same size.
# Ppeprocessing 2 - Normalize - The CNN works better with values between 0 and 1.
# Returns a DataFrame with the images and their labels.

def read_imgs(path: Path, img_width: int, img_height: int) -> pd.DataFrame:
    data = []
    labels = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(root, file))
                img = cv2.resize(img, (img_width, img_height))  # Resize
                img = img / 255.0  # Normalize
                data.append(img)
                class_name = os.path.basename(root)
                labels.append(class_name)
    # Convert labels to integers to use with the CNN.
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    df = pd.DataFrame(list(zip(data, labels)), columns=['image', 'class'])
    return df


def load_data(file_path, img_width, img_height):
    imgs_set = read_imgs(file_path, img_width, img_height)
    return np.stack(imgs_set["image"].values), imgs_set["class"].values
