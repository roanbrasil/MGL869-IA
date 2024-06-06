#Entraînement du modèle
import datetime
import pytz
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path =  ''#mettre ici chemin vers dossier train
validation_path =  ''#mettre ici chemin vers dossier validation
test_path = ''#mettre ici chemin vers dossier test

img_width = 50
img_height = 50
batch_size = 224
epoch_num=12

# Création des générateurs d'images avec augmentation des données
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

# Préparation des générateurs
train_generator = train_datagen.flow_from_directory(
    train_path,
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

# Modèle - CNN
# ------------------------------
hiperparameters = {
    'batch_size': batch_size,
    'epochs': epoch_num
}

# Architecture du modèle CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_width, img_height, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
a = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(a)
# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=hiperparameters['epochs']
)
b = datetime.datetime.now(pytz.timezone("America/Montreal"))
print(b)

print(f"Training time: {b - a}")
# Évaluer le modèle
loss, accuracy = model.evaluate(test_generator)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Sauvegarder le modèle entraîné
model_save_path = "/content/saved_model"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Visualiser les courbes de perte et de précision
plt.figure(figsize=(12, 4))

# Courbe de perte (loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

# Courbe de précision (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()
