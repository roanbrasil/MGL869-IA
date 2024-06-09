import optuna
import os
import datetime
import pytz
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Rescaling,
    Input,
)

# Installation bibliothèque suivantes
#!pip install --upgrade pip
#!pip install optuna
#!pip install tensorflow
#!pip install matplotlib

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--files_root", action="store", dest="files_root", type=str)
args = arg_parser.parse_args()

model_module_path = Path(os.path.realpath(__file__)).parent
file_root = Path(args.files_root)

# Définir les chemins d'accès pour les ensembles d'entraînement et de test
train_path = file_root / "seg_train"  # mettre ici chemin vers dossier train
validation_path = file_root / "seg_test"  # mettre ici chemin vers dossier validation

img_width = 50
img_height = 50


def create_model(trial):
    # Hyperparamètres à tester
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    epochs = trial.suggest_int("epochs", 5, 20)

    # Préparation des générateurs
    train_generator = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode="int",
    )

    validation_generator = tf.keras.utils.image_dataset_from_directory(
        validation_path,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        label_mode="int",
    )

    if len(train_generator) == 0 or len(validation_generator) == 0:
        raise ValueError(
            "Les générateurs d'images sont vides. Vérifiez les chemins d'accès aux répertoires d'images."
        )

    model = tf.keras.models.Sequential(
        [
            Input(shape=(img_width, img_height, 3)),
            Rescaling(1.0 / 255),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(6, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Entraînement modèle et mesurer le temps d'entraînement
    start_train_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
    )
    end_train_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    train_time = end_train_time - start_train_time

    # Évaluation le modèle sur l'ensemble de test et mesure le temps d'inférence
    start_inference_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    end_inference_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    inference_time = end_inference_time - start_inference_time

    trial.set_user_attr("train_time", train_time)
    trial.set_user_attr("inference_time", inference_time)

    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(create_model, n_trials=10)

# Affichage les résultats des essais
for trial in study.trials:
    print(f"Essai {trial.number}:")
    print(f"  Hyperparamètres: {trial.params}")
    print(f"  Précision: {trial.value}")
    print(f"  Temps d'entraînement: {trial.user_attrs['train_time']}")
    print(f"  Temps d'inférence: {trial.user_attrs['inference_time']}")

print(f"Meilleurs hyperparamètres : {study.best_params}")
print(f"Meilleure précision : {study.best_value}")
