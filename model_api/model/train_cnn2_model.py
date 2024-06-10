# Entraînement du modèle
import argparse
import datetime
import pytz
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Input,
)
import matplotlib
import matplotlib.pyplot as plt
from generate_datasets import load_data

matplotlib.use("Qt5Agg")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--files_root", action="store", dest="files_root", type=str)
arg_parser.add_argument("--model_name", action="store", dest="model_name", type=str)
args = arg_parser.parse_args()

model_module_path = Path(os.path.realpath(__file__)).parent
file_root = Path(args.files_root)

# Définir les chemins d'accès pour les ensembles d'entraînement et de test
train_path = file_root / "seg_train"  # mettre ici chemin vers dossier train
validation_path = file_root / "seg_test"  # mettre ici chemin vers dossier validation
test_path = file_root / "seg_pred"  # mettre ici chemin vers dossier test

img_width = 50
img_height = 50
batch_size = 256
epoch_num = 14

# Préparation des datasets
train_x, train_y = load_data(train_path, img_width, img_height)
valid_x, valid_y = load_data(train_path, img_width, img_height)

# Modèle - CNN
# ------------------------------
hiperparameters = {"batch_size": batch_size, "epochs": epoch_num}

# Architecture du modèle CNN
model = tf.keras.models.Sequential(
    [
        Input(shape=(img_width, img_height, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation="relu"),  # Réduction du nombre de neurones
        Dense(6, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
a = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(a)
# Entraîner le modèle
history = model.fit(
    x=train_x,
    y=train_y,
    validation_data=(valid_x, valid_y),
    batch_size=batch_size,
    epochs=hiperparameters["epochs"],
)
b = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(b)

print(f"Training time: {b - a}")
# Évaluer le modèle
loss, accuracy = model.evaluate(x=valid_x, y=valid_y)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Sauvegarder le modèle entraîné
model_save_path = model_module_path / f"repository/{args.model_name}.keras"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Visualiser les courbes de perte et de précision
plt.figure(figsize=(12, 4))

# Courbe de perte (loss)
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Epochs")

# Courbe de précision (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over Epochs")

plt.show()
