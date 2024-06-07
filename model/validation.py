#Installation
#!pip install opendatasets --quiet
#!pip install tensorflow --quiet
#!pip install tfx
#!pip install tensorflow_data_validation


#Conversion dataset vers TFRecords
#Define the data download location and the location to save TFRecord files
import os
import opendatasets as od
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

import os
import tensorflow as tf
from pathlib import Path
from PIL import Image

# Définir les chemins d'accès
train_dir = '' # compléter chemin accès 
validation_dir = ''  # compléter chemin accès 
tfrecords_dir = '/content/tfrecords'    

os.makedirs(tfrecords_dir, exist_ok=True)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label, width, height):
    feature = {
        'image': _bytes_feature(image_string),
        'label': _int64_feature(label),
        'width': _int64_feature(width),
        'height': _int64_feature(height),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecord(image_dir, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    
    for image_path in Path(image_dir).rglob('*.jpg'):
        image = Image.open(image_path)
        image = image.resize((50, 50))  # Redimensionner les images
        image_string = tf.image.encode_jpeg(tf.convert_to_tensor(image))
        label = int(image_path.parent.name)  # Supposant que le nom du dossier parent est l'étiquette
        tf_example = image_example(image_string, label, image.width, image.height)
        writer.write(tf_example.SerializeToString())
    
    writer.close()

convert_to_tfrecord(os.path.join(train_dir, 'train'), os.path.join(tfrecords_dir, 'train.tfrecord'))
convert_to_tfrecord(os.path.join(validation_dir, 'validation'), os.path.join(tfrecords_dir, 'validation.tfrecord'))

print("Conversion terminée !")



#Génération du schéma
tfrecord_path = '/content/tfrecords/train.tfrecord'

import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

stats = tfdv.generate_statistics_from_tfrecord(data_location=tfrecord_path)

schema = tfdv.infer_schema(stats)

# width 
width_feature = schema_pb2.Feature()
width_feature.name = 'width'
width_feature.type = schema_pb2.INT
width_feature.int_domain.min = 50
width_feature.int_domain.max = 50
width_feature.presence.min_count = 1
width_feature.presence.min_fraction = 1.0

# height 
height_feature = schema_pb2.Feature()
height_feature.name = 'height'
height_feature.type = schema_pb2.INT
height_feature.int_domain.min = 50
height_feature.int_domain.max = 50
height_feature.presence.min_count = 1
height_feature.presence.min_fraction = 1.0

# label 
label_feature = schema_pb2.Feature()
label_feature.name = 'label'
label_feature.type = schema_pb2.INT
label_feature.int_domain.min = 0
label_feature.int_domain.max = 5
label_feature.int_domain.is_categorical = True
label_feature.presence.min_count = 1
label_feature.presence.min_fraction = 1.0

# image 
image_feature = schema_pb2.Feature()
image_feature.name = 'image'
image_feature.type = schema_pb2.BYTES
image_feature.presence.min_count = 1
image_feature.presence.min_fraction = 1.0


schema.feature.add().CopyFrom(width_feature)
schema.feature.add().CopyFrom(height_feature)
schema.feature.add().CopyFrom(label_feature)
schema.feature.add().CopyFrom(image_feature)

tfdv.display_schema(schema)

#Validation des données avec la taille de l'image
anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
tfdv.display_anomalies(anomalies)