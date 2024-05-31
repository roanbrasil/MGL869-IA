import datetime

from django.core.management.base import BaseCommand
from images.models import Image, CATEGORIES, PROCESSES
from django.conf import settings

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pickle import dump


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
            len_images = img_width * img_width
            for image in images:
                img = cv2.imread(image.src.path)
                img = cv2.resize(img, (img_width, img_height))  # Resize
                img = img / 255.0  # Normalize
                flat_img = img.flatten()
                data.append(flat_img)
                class_name = CATEGORIES(image.category).label
                labels.append(class_name)
            # Convert labels to integers to use with the CNN.
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            columns = []
            for i in range(len_images):
                columns.append(f"red_{i}")
                columns.append(f"green_{i}")
                columns.append(f"blue_{i}")
            df = pd.DataFrame(data, columns=columns)
            labels = np.atleast_2d(labels).T
            labels_df = pd.DataFrame(labels, columns=["class"])
            df = df.join(labels_df)
            return df

        img_width = 50
        img_height = 50

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
        print(train_data.iloc[:, :-1].shape)
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

        # LinearSVC model
        clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))

        # Convert the 'image' column to a NumPy array
        train_images = train_data.iloc[:, :-1]
        train_labels = train_data["class"].values

        valid_images = valid_data.iloc[:, :-1]
        valid_labels = valid_data["class"].values

        a = datetime.datetime.now()
        print(a)

        # Fit the model
        clf.fit(train_images, train_labels)

        b = datetime.datetime.now()
        print(b)

        print(f"Training time: {b - a}")

        # Evaluate the model
        score = clf.score(valid_images, valid_labels)

        print("LinearSVC Score:")
        print("-------------------")
        print(score)
        print("-------------------")

        # # Save the model in .pkl format
        model_name = options.get("model_name")
        # model.save(settings.BASE_DIR / "model/repository" / f"{model_name}.keras")
        with open(settings.BASE_DIR / "model/repository" / f"{model_name}.pkl", "wb") as f:
            dump(clf, f, protocol=5)
