import cv2
import numpy as np
import pandas as pd
import keras
from pathlib import Path
from django.conf import settings
from sklearn.preprocessing import LabelEncoder

from images.models import Image, CATEGORIES

model_name = "test_model"

model_src = Path(settings.BASE_DIR) / f"model/repository/{model_name}.keras"

model = keras.saving.load_model(str(model_src))


def read_images(images: list[Image], img_width: int, img_height: int) -> np.ndarray:
    data = []
    labels = []
    for image in images:
        img = cv2.imread(image.src.path)
        img = cv2.resize(img, (img_width, img_height))  # Resize
        img = img / 255.0  # Normalize
        data.append(img)
        class_name = 0
        labels.append(class_name)
    # Convert labels to integers to use with the CNN.
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    df = pd.DataFrame(list(zip(data, labels)), columns=["image", "class"])
    return np.stack(df["image"].values)


def classify(images: list[Image]) -> None:
    test_images = read_images(images, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
    classes = model(test_images)
    for i, category in enumerate(np.argmax(classes, axis=-1)):
        images[i].category = CATEGORIES(category)
        images[i].save()
