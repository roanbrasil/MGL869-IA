import optuna
import datetime
import pytz
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Installation bibliothèque suivantes
#!pip install --upgrade pip
#!pip install optuna
#!pip install tensorflow
#!pip install matplotlib

train_path =  ''#mettre ici chemin vers dossier train
validation_path =  ''#mettre ici chemin vers dossier validation
test_path = ''#mettre ici chemin vers dossier test


img_width = 50
img_height = 50

def create_model(trial):
    # Hyperparamètres à tester
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    epochs = trial.suggest_int('epochs', 5, 20)

    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='sparse'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='sparse'
    )

    if train_generator.samples == 0 or validation_generator.samples == 0 or test_generator.samples == 0:
        raise ValueError("Les générateurs d'images sont vides. Vérifiez les chemins d'accès aux répertoires d'images.")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entraînement modèle et mesurer le temps d'entraînement
    start_train_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        verbose=0
    )
    end_train_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    train_time = end_train_time - start_train_time

    # Évaluation le modèle sur l'ensemble de test et mesure le temps d'inférence
    start_inference_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    end_inference_time = datetime.datetime.now(pytz.timezone("America/Montreal"))
    inference_time = end_inference_time - start_inference_time

    trial.set_user_attr('train_time', train_time)
    trial.set_user_attr('inference_time', inference_time)

    return accuracy

study = optuna.create_study(direction='maximize')
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