# Temps d'inférence
import datetime
import pytz
import os
import numpy as np
from pathlib import Path
import cv2
import keras

model_module_path = Path(os.path.realpath(__file__)).parent

img_path = model_module_path.parent / "tests/fixtures/street.jpg"

img_width = 50
img_height = 50
batch_size = 96
epoch_num = 19


def read_images(images: list[bytes], img_width: int, img_height: int) -> np.ndarray:
    data = []
    for image in images:
        img = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))  # Resize
        img = img / 255.0  # Normalize
        data.append(img)
    return np.array(data)


# Préparation des datasets
img_dataset = read_images([img_path.read_bytes()], img_width, img_height)

model_name = "cnn1_model_50x50_batch_96_epochs_19"

classification_module_path = os.path.realpath(__file__)

model_src = model_module_path / f"repository/{model_name}.keras"

model = keras.saving.load_model(str(model_src))

# Mésurer le temps d'inférence
a = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(a)
classes = model(img_dataset)
b = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(b)

print(f"Inference time: {b - a}")
