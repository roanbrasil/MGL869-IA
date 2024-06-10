# Temps d'inférence
import datetime
import pytz
import os
import numpy as np
from pathlib import Path
import cv2
from pickle import load

model_module_path = Path(os.path.realpath(__file__)).parent

img_path = model_module_path.parent / "tests/fixtures/street.jpg"

img_width = 10
img_height = 10


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
img_dataset = np.reshape(img_dataset, (len(img_dataset), img_width*img_height*3))

model_name = "test_model_linear_svc_10x10"

classification_module_path = os.path.realpath(__file__)

model_src = model_module_path / f"repository/{model_name}.pkl"

with open(model_src, "rb") as f:
    clf = load(f)

# Mésurer le temps d'inférence
a = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(a)
classes = clf.predict(img_dataset)
b = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(b)

print(f"Inference time: {b - a}")
