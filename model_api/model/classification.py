import os
import cv2
import numpy as np
import keras
from pathlib import Path

model_name = "cnn2_model_50x50_batch_96_epochs_15"

classification_module_path = os.path.realpath(__file__)

model_src = Path(classification_module_path).parent / f"repository/{model_name}.keras"

model = keras.saving.load_model(str(model_src))

IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50


def read_images(images: list[bytes], img_width: int, img_height: int) -> np.ndarray:
    data = []
    for image in images:
        img = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))  # Resize
        img = img / 255.0  # Normalize
        data.append(img)
    return np.array(data)


def classify(images: list[bytes]) -> list[int]:
    classes = model(read_images(images, IMAGE_WIDTH, IMAGE_HEIGHT))
    categories = []
    for i, category in enumerate(np.argmax(classes, axis=-1)):
        categories.append(int(category))
    return categories
