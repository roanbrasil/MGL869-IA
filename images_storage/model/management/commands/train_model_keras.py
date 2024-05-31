from django.core.management.base import BaseCommand
from images.models import Image, CATEGORIES, PROCESSES
from django.conf import settings

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class Command(BaseCommand):
    help = "Train the model"

    def add_arguments(self, parser):
        parser.add_argument("--model_name", action="store", dest="model_name", type=str)

    def handle(self, *args, **options):
        # Load data and preprocess.
        # ------------------------------
        # Preprocessing 1 - Resize - The CNN requires the images to be in the same size.
        # Ppeprocessing 2 - Normalize - The CNN works better with values between 0 and 1.
        # Returns a DataFrame with the images and their labels.
        def read_imgs(
            process: PROCESSES, img_width: int, img_height: int
        ) -> pd.DataFrame:
            data = []
            labels = []
            images = Image.objects.filter(process=process)
            for image in images:
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

        img_width = settings.IMAGE_WIDTH
        img_height = settings.IMAGE_HEIGHT

        def load_train_data():
            train_imgs = read_imgs(PROCESSES.TRAINING, img_width, img_height)
            return train_imgs

        def load_valid_data():
            valid_imgs = read_imgs(PROCESSES.VALIDATION, img_width, img_height)
            return valid_imgs

        train_data = load_train_data()
        valid_data = load_valid_data()

        # Data analysis
        # ------------------------------
        print("Data shape:")
        print("-------------------")
        print(train_data["image"].shape)
        print("-------------------")

        # print train data first 5 rows
        print("Train data first 5 rows:")
        print("-------------------")
        print(train_data.head())
        print("-------------------")

        # print validation data first 5 rows
        print("Validation data first 5 rows:")
        print("-------------------")

        # Quantify the data
        print("Train data class distribution:")
        print("-------------------")
        print(train_data["class"].value_counts())
        print("-------------------")

        print("Validation data class distribution:")
        print("-------------------")
        print(valid_data["class"].value_counts())
        print("-------------------")

        # Model - CNN
        # ------------------------------
        hiperparameters = {"batch_size": 32, "epochs": 5}

        # The CNN model has the following architecture:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(img_width, img_height, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(6, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Convert the 'image' column to a NumPy array
        train_images = np.stack(train_data["image"].values)
        train_labels = train_data["class"].values

        valid_images = np.stack(valid_data["image"].values)
        valid_labels = valid_data["class"].values

        # Fit the model
        model.fit(
            train_images,
            train_labels,
            batch_size=hiperparameters["batch_size"],
            epochs=hiperparameters["epochs"],
        )

        # Evaluate the model
        model.evaluate(valid_images, valid_labels)

        # Save the model in .keras format
        model_name = options.get("model_name")
        model.save(settings.BASE_DIR / "model/repository" / f"{model_name}.keras")
