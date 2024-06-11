import tensorflow as tf
import os
from pathlib import Path
from generate_datasets import load_data

model_module_path = Path(os.path.realpath(__file__)).parent
model_name = "cnn2_model_50x50_batch_96_epochs_15"
model_src = model_module_path / f"repository/{model_name}.keras"


# Définir les chemins d'accès pour l'ensemble de test
min_test_dir = model_module_path / "repository/minimisation_test_labeled/"

img_width, img_height = 50, 50


test_x, test_y = load_data(min_test_dir, img_width, img_height)

# Evaluation du modèle
model = tf.keras.models.load_model(str(model_src))

# Evaluation du dataset original
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)

print(f"Minimisation test loss: {loss:.4f}")
print(f"Minimisation test accuracy: {accuracy:.4f}")
