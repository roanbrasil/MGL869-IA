# Installation
#!pip install opendatasets --quiet
#!pip install tensorflow --quiet
#!pip install tfx
#!pip install tensorflow_data_validation


# Conversion dataset vers TFRecords
# Define the data download location and the location to save TFRecord files
import os
import argparse
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

import tensorflow as tf
from pathlib import Path
from PIL import Image


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--files_root", action="store", dest="files_root", type=str)
args = arg_parser.parse_args()

model_module_path = Path(os.path.realpath(__file__)).parent
file_root = Path(args.files_root)

# Définir les chemins d'accès pour les ensembles d'entraînement et de test
train_dir = file_root / "seg_train"  # mettre ici chemin vers dossier train
validation_dir = file_root / "seg_test"  # mettre ici chemin vers dossier validation
tfrecords_dir = model_module_path / "repository/content/tfrecords"

os.makedirs(tfrecords_dir, exist_ok=True)

image_width = 50
image_height = 50


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label, width, height):
    feature = {
        "image": _bytes_feature(image_string),
        "label": _int64_feature(label),
        "width": _int64_feature(width),
        "height": _int64_feature(height),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_to_tfrecord(image_dir, output_file):
    writer = tf.io.TFRecordWriter(output_file)

    for image_path in Path(image_dir).rglob("*.jpg"):
        image = Image.open(image_path)
        image = image.resize((image_width, image_height))  # Redimensionner les images
        image_string = tf.image.encode_jpeg(tf.convert_to_tensor(image))
        label = int(
            image_path.parent.name
        )  # Supposant que le nom du dossier parent est l'étiquette
        tf_example = image_example(image_string, label, image.width, image.height)
        writer.write(tf_example.SerializeToString())

    writer.close()


convert_to_tfrecord(
    os.path.join(train_dir, "train"), os.path.join(tfrecords_dir, "train.tfrecord")
)
convert_to_tfrecord(
    os.path.join(validation_dir, "validation"),
    os.path.join(tfrecords_dir, "validation.tfrecord"),
)

print("Conversion terminée !")


# Génération du schéma
tfrecord_path = tfrecords_dir / "train.tfrecord"

stats = tfdv.generate_statistics_from_tfrecord(data_location=str(tfrecord_path))

schema = tfdv.infer_schema(stats)


width_feature = schema_pb2.Feature()
width_feature.name = "width"
width_feature.type = schema_pb2.INT
width_feature.int_domain.min = image_width
width_feature.int_domain.max = image_width


height_feature = schema_pb2.Feature()
height_feature.name = "height"
height_feature.type = schema_pb2.INT
height_feature.int_domain.min = image_height
height_feature.int_domain.max = image_height


schema.feature.add().CopyFrom(width_feature)
schema.feature.add().CopyFrom(height_feature)

tfdv.display_schema(schema)


# Validation des données avec la taille de l'image
anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
tfdv.display_anomalies(anomalies)
