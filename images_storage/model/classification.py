import cv2
import numpy as np
import pandas as pd
# import tensorflow as tf
import keras
from pathlib import Path
from django.conf import settings
from sklearn.preprocessing import LabelEncoder

from images.models import Image, PROCESSES, CATEGORIES

model_name = "test_model"

model_src = Path(settings.BASE_DIR) / f"model/repository/{model_name}.keras"

model = keras.saving.load_model(str(model_src))


def read_images(images: list[Image], img_width: int, img_height: int) -> np.ndarray:
    data = []
    for image in images:
        img = cv2.imread(image.src.path)
        img = cv2.resize(img, (img_width, img_height))  # Resize
        img = img / 255.0  # Normalize
        data.append(img)
    df = pd.DataFrame(data, columns=["image"])
    return np.stack(df.values)


def classify(images: list[Image]) -> None:
    test_images = read_images(images, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
    classes = model(test_images)


"""
from pickle import load
with open("filename.pkl", "rb") as f:
    clf = load(f)
"""