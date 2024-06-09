import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.cluster import KMeans
import random


test_dir = test_path = ""


img_width, img_height = 50, 50

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

def load_data(generator):
    data = []
    labels = []
    for i in range(len(generator)):
        img, label = generator[i]
        data.append(img[0])
        labels.append(label[0])
    return np.array(data), np.array(labels)

test_images, test_labels = load_data(test_generator)

#Clustering
flat_test_images = test_images.reshape(len(test_images), -1)

#KMeans 
num_clusters = 10  # You can adjust this number
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(flat_test_images)
clusters = kmeans.predict(flat_test_images)

#Selection d'échantillon de chaque cluster en fonction de la taille du cluster
sampled_test_images = []
sampled_test_labels = []

for cluster in range(num_clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    sample_size = max(1, len(cluster_indices) // 10)  
    sampled_indices = random.sample(list(cluster_indices), sample_size)
    
    sampled_test_images.extend(test_images[sampled_indices])
    sampled_test_labels.extend(test_labels[sampled_indices])

sampled_test_images = np.array(sampled_test_images)
sampled_test_labels = np.array(sampled_test_labels)

#Evaluation du modèle
model = tf.keras.models.load_model('/path/to/your/model')

# Evaluation du dataset original
loss, original_accuracy = model.evaluate(test_generator, verbose=0)

# Evaluation du dataset minimisé
sampled_test_datagen = tf.data.Dataset.from_tensor_slices((sampled_test_images, sampled_test_labels)).batch(32)
sampled_loss, sampled_accuracy = model.evaluate(sampled_test_datagen, verbose=0)

print(f'Original Test Accuracy: {original_accuracy}')
print(f'Sampled Test Accuracy: {sampled_accuracy}')

