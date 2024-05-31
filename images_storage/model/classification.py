import cv2
import pandas as pd
import tensorflow as tf
from pathlib import Path
from django.conf import settings
from sklearn.preprocessing import LabelEncoder

from images.models import Image, PROCESSES, CATEGORIES

model_name = "test_model"

model_src = Path(settings.BASE_DIR) / f"model/repository/{model_name}.keras"

model = tf.keras.saving.load_model(str(model_src))


def read_img(image: Image, img_width: int, img_height: int) -> pd.DataFrame:
    data = []
    labels = []
    img = cv2.imread(image.src.path)
    img = cv2.resize(img, (img_width, img_height))  # Resize
    img = img / 255.0  # Normalize
    data.append(img)
    class_name = CATEGORIES(image.category).label
    labels.append(class_name)
    # Convert labels to integers to use with the CNN.
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    df = pd.DataFrame(list(zip(data, labels)), columns=["image", "class"])
    return df


def classify(image) -> None:
    Image


"""
from pickle import load
with open("filename.pkl", "rb") as f:
    clf = load(f)
"""